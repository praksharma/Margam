"""Independent implementation of Webb/Jerad/Cartis, arXiv:2607.02194v1.

Algorithms 1–5: blocked residual Jacobians, CountSketch/SRCT, SVD trust-region
steps, monotone PCHIP probes, residual balancing and two-stage ratio selection.
No upstream source is copied. See docs/dsgnar.md for resolved ambiguities.
"""
from copy import deepcopy
import math
import os
import torch
from .pinn import PINN
from ..components.dsgnar_config import dsgnar_defaults,validate_dsgnar_settings


def cosine_transform(x,inverse=False):
    """Orthonormal DCT-II / transpose via FFT; no dense parameter-square matrix."""
    n=x.shape[-1];k=torch.arange(n,device=x.device,dtype=x.dtype)
    phase=torch.polar(torch.ones_like(k),math.pi*k/(2*n))
    scale=torch.full_like(k,1/math.sqrt(2*n));scale[0]=1/math.sqrt(4*n)
    if inverse:
        spectrum=torch.cat((x/scale*phase,torch.zeros_like(x[...,:1],dtype=phase.dtype)),dim=-1)
        return torch.fft.irfft(spectrum,n=2*n,dim=-1)[...,:n]
    spectrum=torch.fft.rfft(torch.cat((x,x.flip(-1)),dim=-1),dim=-1)[...,:n]
    return (spectrum*phase.conj()).real*scale


def right_sketch(x,operators):
    signs,permutation,indices=operators
    return cosine_transform((x*signs)[...,permutation])[...,indices]


def lift_step(y,operators):
    signs,permutation,indices=operators
    full=y.new_zeros(len(signs));full[indices]=y
    transformed=cosine_transform(full,inverse=True)
    result=torch.empty_like(full);result[permutation]=transformed
    return result*signs


def count_add(matrix,residual,jacobian,values,buckets,signs):
    """Deterministic grouped sums, including repeated buckets, on CPU and CUDA."""
    for column in range(buckets.shape[0]):
        order=torch.argsort(buckets[column],stable=True)
        counts=torch.bincount(buckets[column],minlength=len(residual))
        factor=signs[column,order]/math.sqrt(buckets.shape[0])
        matrix+=torch.segment_reduce(jacobian[order]*factor[:,None],'sum',lengths=counts)
        residual+=torch.segment_reduce(values[order]*factor,'sum',lengths=counts)


def lm_step(singular,projected,vt,radius):
    """Safeguarded secular solve, including rank-zero and unconstrained cases."""
    squares=singular.square();g=singular*projected
    active=singular>0
    def coefficients(lam):
        denom=squares+lam
        return torch.where(active,g/denom.clamp_min(torch.finfo(singular.dtype).tiny),0.)
    coeff=coefficients(0.)
    if torch.linalg.vector_norm(coeff).item()<=radius:return -(vt.T@coeff),0.
    lo=0.;hi=max(torch.linalg.vector_norm(g).item()/radius,torch.finfo(singular.dtype).tiny)
    for _ in range(80):
        mid=(lo+hi)/2
        if torch.linalg.vector_norm(coefficients(mid)).item()>radius:lo=mid
        else:hi=mid
    return -(vt.T@coefficients(hi)),hi


def pchip_value(x,y,query):
    """Shape-preserving cubic Hermite interpolation on strictly increasing x."""
    h=x[1:]-x[:-1];slopes=(y[1:]-y[:-1])/h
    d=torch.zeros_like(y)
    if len(x)==2:d[:]=slopes[0]
    else:
        for i in range(1,len(x)-1):
            a,b=slopes[i-1],slopes[i]
            if (a*b).item()>0:
                w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1]
                d[i]=(w1+w2)/(w1/a+w2/b)
        for endpoint,a,b,ha,hb in [(0,slopes[0],slopes[1],h[0],h[1]),(-1,slopes[-1],slopes[-2],h[-1],h[-2])]:
            value=((2*ha+hb)*a-ha*b)/(ha+hb)
            if (value*a).item()<=0:value=a*0
            elif (a*b).item()<0 and abs(value)>abs(3*a):value=3*a
            d[endpoint]=value
    index=max(0,min(len(x)-2,int(torch.searchsorted(x,x.new_tensor(query)).item())-1))
    t=(query-x[index])/h[index]
    return ((2*t**3-3*t**2+1)*y[index]+(t**3-2*t**2+t)*h[index]*d[index]
            +(-2*t**3+3*t**2)*y[index+1]+(t**3-t**2)*h[index]*d[index+1]).item()


def choose_radius(radii,ratios,target):
    # Fit ratio as a function of log radius: unlike inverse interpolation, this
    # remains well-defined on equal-ratio plateaus, including clipped negatives.
    safe=torch.cummin(torch.as_tensor(ratios,dtype=torch.float64).clamp_min(0),0).values
    if safe[-1]>=target:return radii[-1]
    if safe[0]<=target:return radii[0]
    knots=torch.tensor([math.log(radius) for radius in radii],dtype=torch.float64)
    low=knots[0].item();high=knots[-1].item()
    for _ in range(60):
        middle=(low+high)/2
        if pchip_value(knots,safe,middle)>=target:low=middle
        else:high=middle
    return math.exp((low+high)/2)


def reweight(weights,losses,alpha):
    if alpha==0 or sum(losses.values())==0:return dict(weights)
    average=sum(losses.values())/len(losses)
    logs={name:math.log(value)+alpha*math.log(average/(losses[name]+1e-8)) for name,value in weights.items()}
    maximum=max(logs.values());values={name:math.exp(max(-700.,value-maximum)) for name,value in logs.items()}
    mean=sum(values.values())/len(values)
    return {name:value/mean for name,value in values.items()}


def next_ratio(history,settings,current):
    count=settings['history_window']
    if current==settings['stage2_ratio'] or len(history)<count or len(history)<=settings.get('stage_grace',0):return current
    y=torch.tensor([math.log(max(v,1e-30)) for v in history[-count:]],dtype=torch.float64)
    x=torch.arange(count,dtype=y.dtype);x=x-x.mean();y=y-y.mean()
    slope=(x@y)/(x@x);denom=torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y)
    correlation=(x@y)/denom if denom>0 else y.new_tensor(0.)
    return settings['stage2_ratio'] if slope>1e-4 and correlation>.1 else current


class DSGNARPINN(PINN):
    """Fixed-point least-squares trainer; train budget counts attempted steps."""
    def __init__(self,problem,network,*,weights=None,settings=None,configuration=None):
        super().__init__(problem,network,weights=weights)
        self.configuration=deepcopy(configuration or {})
        for key in ('iterations','device'):self.configuration.pop(key,None)
        self.settings=deepcopy(dsgnar_defaults() if settings is None else settings)
        self.settings.setdefault('stage_grace',0)
        if isinstance(self.configuration.get('dsgnar_settings'),dict):self.configuration['dsgnar_settings'].setdefault('stage_grace',0)
        validate_dsgnar_settings(self.settings)

    def compile(self,optimiser_string='adam',lr=.001,metrics_string='MSE',device='auto',*,optimiser=None):
        if optimiser is not None or metrics_string!='MSE' or optimiser_string!='adam':raise ValueError('DSGNAR owns its SVD trust-region update; custom optimizers are not supported.')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        self.network.to(dtype=getattr(torch,self.settings['precision']))
        # Reuse shared data/shape validation. Adam is an empty compatibility
        # payload only: no Adam step is ever taken by this trainer.
        super().compile(lr=lr,device=device)
        self.parameters=[p for p in self.network.parameters() if p.requires_grad]
        if any(p.dtype!=self.dtype for p in self.parameters):raise ValueError('DSGNAR requires a uniform scalar dtype.')
        self.names=['pde',*(c.name for c,_,_ in self._constraints)]
        self.initial_weights={name:self.weights.get(name,1.) for name in self.names}
        if any(value<=0 for value in self.initial_weights.values()):raise ValueError('DSGNAR requires positive condition weights.')
        self.weights=deepcopy(self.initial_weights)
        self.generator=torch.Generator(device='cpu').manual_seed(self.settings['seed'])
        self.radius=self.settings['initial_radius'];self.target_ratio=self.settings['target_ratio']
        self.diagnostics=[];self.lambda_history=[];self.converged=False
        return self

    def _target(self,source,points):
        # Fixed target data have one canonical CPU evaluation, including when
        # training on CUDA; archive validation must not depend on libm rounding.
        target=source(points.detach().cpu()) if callable(source) else source
        return torch.as_tensor(target,dtype=self.dtype,device=self.device).detach().clone()

    def _conditions(self):
        return [('pde',self.collocation_point_sample,self.collocation_point_labels,None),
                *((c.name,p,t,c.operator) for c,p,t in self._constraints)]

    def _batches(self,condition,batch_size=None):
        name,points,target,operator=condition
        batch_size=self.settings['batch_size'] if batch_size is None else batch_size
        for start in range(0,len(points),batch_size):
            x=points[start:start+batch_size].detach().requires_grad_(True)
            values=self.network(x)
            values=self.problem.residual(x,values) if name=='pde' else operator(x,values) if operator is not None else values
            batch_target=target if target.ndim==0 else target[start:start+len(x)]
            values,batch_target=self._aligned(values,batch_target,len(x))
            yield (values-batch_target).reshape(-1)

    def _flat(self):return torch.cat([p.detach().reshape(-1) for p in self.parameters])
    def _assign(self,flat):
        with torch.no_grad():
            start=0
            for p in self.parameters:p.copy_(flat[start:start+p.numel()].view_as(p));start+=p.numel()

    def _losses(self):
        result={}
        for condition in self._conditions():
            total=0.;count=0
            for residual in self._batches(condition,batch_size=1024):total+=residual.detach().square().sum().item();count+=residual.numel()
            result[condition[0]]=total/count
        return result

    def _jacobian_batches(self,condition):
        """Bounded generic reverse-mode blocks; concrete research code may override."""
        for values in self._batches(condition):
            if values.requires_grad:
                gradients=torch.autograd.grad(values,self.parameters,grad_outputs=torch.eye(len(values),device=self.device,dtype=self.dtype),
                    is_grads_batched=True,retain_graph=False,allow_unused=True)
                jacobian=torch.cat([(p.new_zeros((len(values),p.numel())) if g is None else g.reshape(len(values),-1)) for p,g in zip(self.parameters,gradients)],dim=1).detach()
            else:jacobian=values.new_zeros((len(values),sum(p.numel() for p in self.parameters)))
            yield jacobian,values.detach()

    def _sketch(self,operators,size):
        matrix=torch.zeros((size,size),device=self.device,dtype=self.dtype);vector=matrix.new_zeros(size)
        losses={}
        for condition in self._conditions():
            total=0.;count=0
            # Each condition can have a different number of rows. Count its scalar
            # components once; no partial batch is dropped or padded.
            for values in self._batches(condition):count+=values.numel()
            for jacobian,values in self._jacobian_batches(condition):
                if not torch.isfinite(values).all():raise FloatingPointError('Non-finite DSGNAR residual.')
                if not torch.isfinite(jacobian).all():raise FloatingPointError('Non-finite DSGNAR Jacobian.')
                compressed=right_sketch(jacobian,operators)
                factors=math.sqrt(self.weights[condition[0]]/count)
                buckets=torch.randint(size,(self.settings['hashes'],len(values)),generator=self.generator).to(self.device)
                signs=(torch.randint(2,buckets.shape,generator=self.generator)*2-1).to(device=self.device,dtype=self.dtype)
                count_add(matrix,vector,compressed*factors,values.detach()*factors,buckets,signs)
                total+=values.detach().square().sum().item()
            losses[condition[0]]=total/count
        return matrix,vector,losses

    def train(self,iterations,display_every=1):
        previous=torch.get_deterministic_debug_mode()
        try:
            torch.use_deterministic_algorithms(True)
            return self._train(iterations,display_every)
        finally:torch.set_deterministic_debug_mode(previous)

    def _train(self,iterations,display_every=1):
        self._require_compiled()
        if type(iterations) is not int or iterations<0 or type(display_every) is not int or display_every<0:raise ValueError('Iterations/display interval must be non-negative integers.')
        if self.sampling_controller is not None or self.loss_formulation!='mean':raise ValueError('DSGNAR currently requires fixed points and mean loss.')
        self.network.train()
        for _ in range(iterations):
            if self.converged:break
            base=self._flat();rng=self.generator.get_state();old_weights=deepcopy(self.weights)
            try:
                dimension=len(base);size=min(self.settings['sketch_size'],dimension)
                operators=((torch.randint(2,(dimension,),generator=self.generator)*2-1).to(device=self.device,dtype=self.dtype),
                           torch.randperm(dimension,generator=self.generator).to(self.device),torch.randperm(dimension,generator=self.generator)[:size].to(self.device))
                matrix,vector,_=self._sketch(operators,size)
                # Actual reductions must compare the same evaluation batching.
                # Jacobian micro-batches can round differently from probe batches,
                # especially near the single-precision accuracy floor.
                terms=self._losses()
                # Tiny CUDA SVDs can vary in their last bits across fresh library
                # contexts. Solve small sketches on CPU; derivatives stay on target.
                solve_matrix=matrix.cpu() if matrix.is_cuda and size<=128 else matrix
                solve_vector=vector.to(solve_matrix.device)
                u,s,vt=torch.linalg.svd(solve_matrix,full_matrices=False,driver='gesvd' if solve_matrix.is_cuda else None);projected=u.T@solve_vector
                before=sum(self.weights[name]*value for name,value in terms.items())
                if not math.isfinite(before):raise FloatingPointError('Non-finite DSGNAR objective.')
                def trial(radius):
                    step,lam=lm_step(s,projected,vt,radius)
                    predicted=(solve_vector.square().sum()-(solve_vector+solve_matrix@step).square().sum()).item()
                    self._assign(base+lift_step(step.to(self.device),operators));losses=self._losses()
                    objective=sum(self.weights[name]*value for name,value in losses.items())
                    ratio=(before-objective)/predicted if predicted>0 and math.isfinite(objective) else -1.
                    if not math.isfinite(ratio):ratio=-1.
                    return objective,ratio,lam,losses,step
                low=self.radius/3;high=min(self.radius*3,self.settings['max_radius'])
                radii=torch.logspace(math.log10(low),math.log10(high),self.settings['probes'],dtype=torch.float64).tolist()
                ratios=[trial(radius)[1] for radius in radii]
                chosen=choose_radius(radii,ratios,self.target_ratio)
                after,ratio,lam,after_terms,step=trial(chosen)
                accepted=math.isfinite(after) and after<before and ratio>0
                self._assign(base+lift_step(step.to(self.device),operators) if accepted else base)
                next_radius=chosen if accepted else self.radius/3
                new_weights=reweight(self.weights,{name:self.weights[name]*value for name,value in terms.items()},self.settings['reweight_exponent'])
                record=dict(terms,total=before)
                self.record_contributions(record,{name:base.new_tensor(value) for name,value in terms.items()},old_weights)
                diagnostic=dict(step=self.iter+1,accepted=accepted,radius=chosen,next_radius=next_radius,
                    regularization=lam,ratio=ratio,target_ratio=self.target_ratio,objective_before=before,
                    objective_after=after if accepted else before,weights=old_weights)
            except Exception:
                self._assign(base);self.generator.set_state(rng);raise
            self.iter+=1;self.radius=next_radius;self.weights=new_weights
            self.lambda_history.append(lam);self.target_ratio=next_ratio(self.lambda_history,self.settings,self.target_ratio)
            self.converged=self.radius<self.settings['min_radius']
            self.diagnostics.append(diagnostic);self.loss_history.append(record);self.training_history.append(before)
            self.PDE_loss=base.new_tensor(terms['pde']);self.BC_loss=base.new_tensor(sum(v for n,v in terms.items() if n!='pde'));self.total_loss=base.new_tensor(before)
            if display_every and self.iter%display_every==0:print(f'Attempt: {self.iter} DSGNAR loss: {before:.6g} accepted: {accepted}')
        return self.training_history

    def dsgnar_summary(self):
        return dict(attempts=self.iter,accepted=sum(row['accepted'] for row in self.diagnostics),radius=self.radius,
                    target_ratio=self.target_ratio,converged=self.converged,weights=deepcopy(self.weights),
                    last=deepcopy(self.diagnostics[-1]) if self.diagnostics else None)

    def training_state(self):
        state=super().training_state()
        state['dsgnar']=deepcopy(dict(version=1,configuration=self.configuration,settings=self.settings,initial_weights=self.initial_weights,weights=self.weights,
            radius=self.radius,target_ratio=self.target_ratio,generator=self.generator.get_state(),diagnostics=self.diagnostics,
            lambda_history=self.lambda_history,converged=self.converged))
        return state

    def load_training_state(self,state):
        from ..components.dsgnar_state import validate_state
        payload=validate_state(state,self.settings,self.initial_weights,self.configuration)
        super().load_training_state({key:value for key,value in state.items() if key!='dsgnar'})
        self.weights=deepcopy(payload['weights']);self.radius=payload['radius'];self.target_ratio=payload['target_ratio']
        self.generator.set_state(payload['generator'].cpu());self.diagnostics=deepcopy(payload['diagnostics'])
        self.lambda_history=list(payload['lambda_history']);self.converged=payload['converged']
        return self
