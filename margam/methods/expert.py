"""Expert's Guide v1: causal residuals, L2-gradient balancing and fresh samples.

Independent PyTorch implementation from the paper equations, not a JAX port.
Concrete scalar and flow workflows; optional NTK diagonal balancing.
"""
from copy import deepcopy
from dataclasses import asdict,replace
import math
import torch
from .pinn import PINN
from ..components.expert_config import validate_expert_problem


class ExpertPINN(PINN):
    def __init__(self,problem,network,*,config,weights=None,loss_formulation='mean',initial_teacher=None):
        super().__init__(problem,network,weights=weights,loss_formulation=loss_formulation)
        self.sample_config=deepcopy(config);self.options=deepcopy(config.expert_settings)
        self.initial_teacher=initial_teacher
        self._expert_fingerprint=None

    def compile(self,optimiser_string='adam',lr=.001,metrics_string='MSE',device='auto',*,optimiser=None):
        validate_expert_problem(self.sample_config)
        if self.loss_formulation!='mean':raise ValueError('Expert PINN currently requires mean MSE.')
        if not self.sample_config.is_paper_case and {condition.name for condition in self.problem.constraints}!={'left','right','initial'}:raise ValueError('Expert PINN requires left, right and initial constraint sets.')
        if any(not callable(source) and torch.as_tensor(source).ndim!=0 for source in (self.problem.residual_target,*(condition.target for condition in self.problem.constraints))):raise ValueError('Fresh Expert samples require scalar or coordinate-callable targets.')
        if optimiser_string!='adam' or metrics_string!='MSE' or optimiser is not None:raise ValueError('Expert PINN uses Adam and MSE.')
        self._expert_fingerprint=None
        super().compile(optimiser_string,lr,metrics_string,device)
        self.initial_lr=lr;self.coefficients=dict(pde=1.,**({'initial':1.} if self.sample_config.is_time_dependent else {}),boundary=1.);self.balance_history=[];self._causal_min=1.
        description=asdict(self.sample_config)
        for key in ('iterations','device'):description.pop(key)
        self.expert_signature=dict(algorithm='expert-guide-v1-ntk-diagonal' if self.options['balance_mode']=='ntk' else 'expert-guide-v1-sum-grad-norm',configuration=description,lr=lr,weights=deepcopy(self.weights))
        if self.initial_teacher is not None:self.expert_signature['initial_teacher']=self.initial_teacher.identity
        self._fresh_samples(0)
        self._expert_fingerprint=self._canonical_fingerprint()
        return self

    def _fresh_samples(self,phase):
        if self.sample_config.is_paper_case:
            points,conditions=self.problem.refresh_samples(phase)
            self.collocation_point_sample=self._points(points);self.collocation_point_labels=self._target(self.problem.residual_target,self.collocation_point_sample)
            self._constraints=[]
            for condition in conditions:
                points=self._points(condition.points);self._constraints.append((condition,points,self._target(condition.target,points)))
            self._transfer_initial()
            return
        from ..templates.poisson import sample_problem
        config=deepcopy(self.sample_config)
        offset=104729*phase
        config.seed=(config.seed+offset)%2**31
        config.boundary_sampling={side:replace(config.boundary_recipe(side),seed=(config.boundary_recipe(side).seed+offset)%2**31) for side in config.sides}
        interior,boundaries=sample_problem(config)
        lower,upper=config.bounds;chunks=self.options['chunks'];per=len(interior)//chunks
        # Equal-width temporal strata, not sorting time independently of space.
        for index,row in enumerate(interior):
            fraction=(row[1]-lower[1])/(upper[1]-lower[1])
            row[1]=lower[1]+(index//per+fraction)*(upper[1]-lower[1])/chunks
        self.collocation_point_sample=self._points(interior)
        self.collocation_point_labels=self._target(self.problem.residual_target,self.collocation_point_sample)
        samples={config.constraint_name(side):rows for side,rows in boundaries.items()}
        self._constraints=[(condition,self._points(samples[condition.name]),self._target(condition.target,self._points(samples[condition.name]))) for condition,_,_ in self._constraints]

        self._transfer_initial()

    def _transfer_initial(self):
        if self.initial_teacher is None:return
        from dataclasses import replace
        self._constraints=[(replace(condition,target=self.initial_teacher),points,self._target(self.initial_teacher,points)) if condition.name=='initial' else (condition,points,target) for condition,points,target in self._constraints]

    def _canonical_fingerprint(self):
        # Evaluate target expressions on CPU for device-independent archive identity.
        # GPU/CPU transcendental implementations can differ by a last bit.
        sample,labels,constraints=self.collocation_point_sample,self.collocation_point_labels,self._constraints
        def target(source,points):
            value=source(points) if callable(source) else source
            return torch.as_tensor(value,dtype=self.dtype,device='cpu').detach()
        try:
            self.collocation_point_sample=sample.detach().cpu()
            self.collocation_point_labels=target(self.problem.residual_target,self.collocation_point_sample)
            self._constraints=[(condition,points.detach().cpu(),target(condition.target,points.detach().cpu())) for condition,points,_ in constraints]
            return super().data_fingerprint()
        finally:
            self.collocation_point_sample,self.collocation_point_labels,self._constraints=sample,labels,constraints

    def data_fingerprint(self):
        return self._expert_fingerprint if self._expert_fingerprint is not None else super().data_fingerprint()

    @staticmethod
    def causal_objective(chunk_losses,epsilon):
        prefix=torch.cat((chunk_losses.new_zeros(1),chunk_losses.detach().cumsum(0)[:-1]))
        weights=torch.exp(-epsilon*prefix)
        return (weights*chunk_losses).mean(),weights

    def compute_losses(self):
        samples=self.loss_samples()
        terms={name:self._term(*sample) for name,sample in samples.items()}
        # Fresh samples are stored in temporal-stratum order, keeping each (x,t) pair intact.
        if not self.sample_config.is_time_dependent:return terms
        sample=samples['pde'];errors=self.point_squared_errors(*sample).flatten()
        chunk_losses=errors.reshape(self.options['chunks'],-1).mean(1)
        terms['pde'],causal=self.causal_objective(chunk_losses,self.options['causal_epsilon'])
        self._causal_min=causal.detach().min().item()
        return terms

    def _groups(self,terms):
        groups=dict(pde=terms['pde']*self.weights.get('pde',1.))
        if 'initial' in terms:groups['initial']=terms['initial']*self.weights.get('initial',1.)
        groups['boundary']=sum(value*self.weights.get(name,1.) for name,value in terms.items() if name not in ('pde','initial'))
        return groups

    @staticmethod
    def ntk_trace(values,parameters,maximum,seed=0):
        """Diagonal Jacobian Gram trace; row subsampling estimates the full trace."""
        if values.ndim==1:values=values[:,None]
        count=len(values);take=min(count,maximum)
        indices=torch.randperm(count,generator=torch.Generator().manual_seed(seed))[:take].to(values.device) if take<count else torch.arange(count,device=values.device)
        trace=0.
        for value in values[indices].flatten():
            gradients=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True) if value.requires_grad else (None,)*len(parameters)
            if any(g is not None and not torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Non-finite NTK diagonal; no update was taken.')
            trace+=sum(g.detach().double().square().sum().item() for g in gradients if g is not None)
        return trace*count/take

    def _ntk_traces(self):
        parameters=tuple(p for p in self.network.parameters() if p.requires_grad)
        traces=dict.fromkeys(self.coefficients,0.)
        for index,(name,(values,_,_)) in enumerate(self.loss_samples().items()):
            group=name if name in ('pde','initial') else 'boundary'
            traces[group]+=self.ntk_trace(values,parameters,self.options['ntk_points'],(self.sample_config.seed+104729*self.iter+7919*index)%2**31)
        return traces

    def _balanced(self,groups):
        parameters=tuple(p for p in self.network.parameters() if p.requires_grad)
        norms={}
        for name,loss in (groups.items() if self.options['balance_mode']=='gradient' else []):
            gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else (None,)*len(parameters)
            if any(g is not None and not torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Non-finite Expert PINN gradient norm; no update was taken.')
            norms[name]=math.sqrt(sum(g.detach().double().square().sum().item() for g in gradients if g is not None))
        if self.options['balance_mode']=='ntk':norms=self._ntk_traces()
        reference=sum(norms.values());next_weights={};targets={}
        for name,old in self.coefficients.items():
            target=None if norms[name]<=self.options['gradient_floor'] else min(reference/norms[name],self.options['weight_cap'])
            targets[name]=target
            next_weights[name]=old if target is None else self.options['momentum']*old+(1-self.options['momentum'])*target
        return next_weights,dict(step=self.iter+1,norms=norms,targets=targets,weights=deepcopy(next_weights))

    def train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:raise ValueError('Iterations/display interval must be non-negative integers.')
        self.network.train()
        for _ in range(iterations):
            self._fresh_samples(self.iter)
            self.optimiser.zero_grad(set_to_none=True)
            terms=self.compute_losses();groups=self._groups(terms)
            if any(not torch.isfinite(value).item() for value in groups.values()):raise FloatingPointError('Non-finite Expert PINN loss; no update was taken.')
            weights=self.coefficients;event=None
            if (self.iter+1)%self.options['balance_every']==0:weights,event=self._balanced(groups)
            total=sum(groups[name]*weights[name] for name in groups)
            if not torch.isfinite(total).item():raise FloatingPointError('Non-finite Expert PINN objective; no update was taken.')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):raise FloatingPointError('Non-finite Expert PINN gradient; no update was taken.')
            rate=self.initial_lr*self.options['decay_rate']**(self.iter/self.options['decay_steps'])
            for group in self.optimiser.param_groups:group['lr']=rate
            self.optimiser.step();self.iter+=1;self.coefficients=weights
            if event is not None:self.balance_history.append(event)
            record={name:value.detach().item() for name,value in terms.items()};record['total']=total.detach().item()
            self.loss_history.append(record);self.training_history.append(record['total'])
            self.PDE_loss=terms['pde'].detach();self.BC_loss=sum(value.detach() for name,value in terms.items() if name!='pde');self.total_loss=total.detach()
            if display_every and self.iter%display_every==0:print(f"Iteration: {self.iter} Expert loss: {record['total']:.6g}")
        return self.training_history

    def expert_summary(self):return dict(step=self.iter,coefficients=deepcopy(self.coefficients),learning_rate=self.optimiser.param_groups[0]['lr'],minimum_causal_weight=self._causal_min,balance_mode=self.options['balance_mode'])

    def training_state(self):
        state=super().training_state()
        state['expert']=deepcopy(dict(settings=self.expert_signature,coefficients=self.coefficients,history=self.balance_history))
        return state

    def load_training_state(self,state):
        self._require_compiled()
        payload=state.get('expert') if isinstance(state,dict) else None
        if isinstance(payload,dict) and isinstance(payload.get('settings'),dict):
            payload=deepcopy(payload)
            from ..components.expert_config import expert_defaults
            configuration=payload['settings'].get('configuration',{})
            if not isinstance(configuration,dict):raise ValueError('Invalid Expert PINN checkpoint configuration.')
            if isinstance(configuration,dict):
                from ..templates.poisson import PoissonConfig
                defaults=asdict(PoissonConfig())
                for key in ('t_min','t_max','cylinder_x','cylinder_y','cylinder_radius','fourth_order','startup_time','scale_settings'):configuration.setdefault(key,defaults[key])
            saved=configuration.get('expert_settings')
            if isinstance(saved,dict):payload['settings']['configuration']['expert_settings']={**expert_defaults(),**saved}
        if not isinstance(payload,dict) or set(payload)!={'settings','coefficients','history'} or payload['settings']!=self.expert_signature:raise ValueError('Expert PINN checkpoint settings differ.')
        steps=state.get('iter');history=payload['history'];names=set(self.coefficients)
        if type(steps) is not int or not 0<=steps<=100000 or not isinstance(history,list) or len(history)!=steps//self.options['balance_every']:raise ValueError('Expert PINN balance history phase differs.')
        previous=dict.fromkeys(names,1.)
        def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
        for index,event in enumerate(history):
            if not isinstance(event,dict) or set(event)!={'step','norms','targets','weights'} or type(event['step']) is not int or event['step']!=(index+1)*self.options['balance_every']:raise ValueError('Invalid Expert PINN balance event.')
            if any(not isinstance(event[key],dict) or set(event[key])!=names for key in ('norms','targets','weights')) or not all(finite(v) for v in event['norms'].values()):raise ValueError('Invalid Expert PINN gradient norms.')
            reference=sum(event['norms'].values())
            for name in names:
                target=None if event['norms'][name]<=self.options['gradient_floor'] else min(reference/event['norms'][name],self.options['weight_cap'])
                if event['targets'][name]!=target:raise ValueError('Invalid Expert PINN gradient ratio.')
                expected=previous[name] if target is None else self.options['momentum']*previous[name]+(1-self.options['momentum'])*target
                if not finite(event['weights'][name]) or not math.isclose(event['weights'][name],expected,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Invalid Expert PINN moving average.')
            previous=event['weights']
        if payload['coefficients']!=previous:raise ValueError('Expert PINN coefficients do not match history.')
        rates=[group['lr'] for group in self.optimiser.param_groups]
        expected_rate=self.initial_lr*self.options['decay_rate']**(max(steps-1,0)/self.options['decay_steps'])
        try:
            for group in self.optimiser.param_groups:group['lr']=expected_rate
            super().load_training_state({key:value for key,value in state.items() if key!='expert'})
        except Exception:
            for group,rate in zip(self.optimiser.param_groups,rates):group['lr']=rate
            raise
        self.coefficients=deepcopy(payload['coefficients']);self.balance_history=deepcopy(history)
        self._fresh_samples(max(steps-1,0))
        return self
