"""Sequential Expert time windows with frozen prediction transfer and safe resume."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import torch


def cpu_state(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return deepcopy(value)


class InitialTeacher:
    def __init__(self,network,components):
        self.network=deepcopy(network).cpu().eval();self.components=components;self.cache={}
        digest=hashlib.sha256()
        for name,value in self.network.state_dict().items():
            digest.update(name.encode());digest.update(value.detach().contiguous().numpy().tobytes())
        self.identity=digest.hexdigest()+':'+str(components)
        for parameter in self.network.parameters():parameter.requires_grad_(False)

    def predict_all(self,points):
        return self._evaluate(points)

    def __call__(self,points):return self._evaluate(points)[:,:self.components]

    def _evaluate(self,points):
        key=str(points.device)
        if key=='cpu':network=self.network
        else:
            if key not in self.cache:self.cache[key]=deepcopy(self.network).to(points.device)
            network=self.cache[key]
        with torch.no_grad():return network(points).detach()


class TimeMarchingPINN:
    def __init__(self,config):
        config.validate()
        if config.method!='expert' or not config.is_time_dependent or config.expert_settings['windows']<2:raise ValueError('Time marching requires Expert training and multiple transient windows.')
        self.config=deepcopy(config);self.completed=[];self.teachers=[]
        self.count=config.expert_settings['windows'];self.window_steps=config.expert_settings['window_steps']
        lower,upper=config.bounds;self.start=lower[-1];self.end=upper[-1]
        self.edges=[self.start+(self.end-self.start)*i/self.count for i in range(self.count+1)]
        self.signature=asdict(config)
        for key in ('iterations','device'):self.signature.pop(key)
        self.active=self._build(0)

    def _build(self,index,teacher=None):
        from ..templates.poisson import build_pinn
        c=deepcopy(self.config);c.expert_settings['windows']=1
        if len(c.bounds[0])==2:c.y_min,c.y_max=self.edges[index:index+2]
        else:c.t_min,c.t_max=self.edges[index:index+2]
        return build_pinn(c,initial_teacher=teacher,time_origin=self.start)

    def __getattr__(self,name):return getattr(self.active,name)

    @property
    def iter(self):return len(self.completed)*self.window_steps+self.active.iter

    @property
    def loss_history(self):return [row for state in self.completed for row in state['loss_history']]+self.active.loss_history

    @property
    def training_history(self):return [row['total'] for row in self.loss_history]

    @property
    def outputs_ready(self):return len(self.completed)==self.count-1 and self.active.iter>0

    def _advance(self):
        state=cpu_state(self.active.training_state())
        # A transient velocity/pressure problem transfers velocity, not pressure ICs.
        components=2 if self.config.pde_preset=='expert_cylinder' else len(self.config.field_names)
        teacher=InitialTeacher(self.active.network,components)
        weights={name:parameter.detach().clone() for name,parameter in self.active.network.named_parameters()}
        self.completed.append(state);self.teachers.append(teacher)
        self.active=self._build(len(self.completed),teacher)
        with torch.no_grad():
            for name,parameter in self.active.network.named_parameters():parameter.copy_(weights[name])
        # Retain the new window's coordinate-scaling buffers; reset Adam/EMA/decay.
        # Teacher identity is immutable; fingerprint depends only on samples/targets.

    def train(self,iterations,display_every=1):
        if type(iterations) is not int or iterations<0:raise ValueError('Updates must be non-negative integers.')
        if type(display_every) is not int or display_every<0:raise ValueError('Display interval must be non-negative.')
        while iterations:
            if len(self.completed)<self.count-1 and self.active.iter>=self.window_steps:self._advance()
            take=iterations if len(self.completed)==self.count-1 else min(iterations,self.window_steps-self.active.iter)
            self.active.train(take,display_every);iterations-=take
        return self.training_history

    def expert_summary(self):
        summary=self.active.expert_summary();summary.update(step=self.iter,window=len(self.completed)+1,windows=self.count,window_step=self.active.iter,trained_time_end=self.edges[len(self.completed)+1],outputs_ready=self.outputs_ready)
        return summary

    def predict(self,points):
        x=self.active._points(points);index=len(self.completed);horizon=self.edges[index+1]
        if not torch.isfinite(x).all() or ((x[:,-1]<self.start)|(x[:,-1]>horizon)).any():raise ValueError('Predictions outside the trained time windows are unavailable.')
        result=x.new_empty((len(x),len(self.config.field_names)))
        for i in range(index+1):
            mask=(x[:,-1]>=self.edges[i] if i==0 else x[:,-1]>self.edges[i])&(x[:,-1]<=self.edges[i+1])
            if not mask.any():continue
            if i==index:result[mask]=self.active.predict(x[mask])
            else:
                # Stored teachers preserve all predicted fields, including pressure.
                result[mask]=self.teachers[i].predict_all(x[mask])
        return result

    def training_state(self):
        state=self.active.training_state();state.update(iter=self.iter,loss_history=deepcopy(self.loss_history),training_history=self.training_history)
        state['marching']=dict(configuration=deepcopy(self.signature),completed=cpu_state(self.completed))
        return state

    def load_training_state(self,state):
        payload=state.get('marching') if isinstance(state,dict) else None
        if isinstance(payload,dict) and isinstance(payload.get('configuration'),dict):
            from ..components.scale_config import scale_defaults
            payload=deepcopy(payload);payload['configuration'].setdefault('scale_settings',scale_defaults())
        if not isinstance(payload,dict) or set(payload)!={'configuration','completed'} or payload['configuration']!=self.signature:raise ValueError('Time-marching configuration differs.')
        completed=payload['completed']
        if not isinstance(completed,list) or len(completed)>=self.count:raise ValueError('Invalid completed time windows.')
        # Validate into an independent candidate before mutating the live run.
        candidate=TimeMarchingPINN(self.config);prefix=[]
        for previous in completed:
            if not isinstance(previous,dict) or previous.get('iter')!=self.window_steps:raise ValueError('Invalid time-window update count.')
            candidate.active.load_training_state(previous);prefix.extend(previous['loss_history']);candidate._advance()
        steps=state.get('iter');offset=len(prefix)
        if type(steps) is not int or not offset<=steps<=100000 or len(state.get('loss_history',[]))!=steps or state['loss_history'][:offset]!=prefix or state.get('training_history')!=[row['total'] for row in state['loss_history']]:raise ValueError('Invalid time-marching history.')
        local=steps-offset
        if len(completed)<self.count-1 and local>self.window_steps:raise ValueError('Time-window phase exceeds its update budget.')
        active={k:v for k,v in state.items() if k!='marching'}
        active.update(iter=local,loss_history=state['loss_history'][offset:],training_history=state['training_history'][offset:])
        candidate.active.load_training_state(active)
        self.completed,self.teachers,self.active=candidate.completed,candidate.teachers,candidate.active
        return self
