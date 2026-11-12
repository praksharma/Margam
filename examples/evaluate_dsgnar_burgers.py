"""Reload a known periodic Burgers benchmark checkpoint without training."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from examples.dsgnar_accuracy import PeriodicMLP,burgers_reference
from margam.research import DSGNARPINN,PINNProblem,PointConstraint,derivative
from margam.execution.codec import load_packet
from margam.methods.dsgnar import next_ratio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--points',type=int,default=32768);parser.add_argument('--initial',type=int,default=16384)
    parser.add_argument('--output',type=Path);parser.add_argument('--csv',type=Path);args=parser.parse_args()
    torch.set_num_threads(1);state=load_packet(args.checkpoint)
    settings=state['dsgnar']['settings'];configuration=state['dsgnar']['configuration']
    known={'periodic-burgers-2607.02194-C1-v3','periodic-burgers-2607.02194-C1-v4',
           'periodic-burgers-2607.02194-C1-v5-forward-probes','periodic-burgers-2607.02194-C1-v6-forward-probes-grace'}
    if configuration.get('case') not in known or configuration.get('viscosity')!=.01:raise ValueError('Not a supported periodic Burgers benchmark checkpoint.')
    width,depth,seed=(configuration[key] for key in ('width','depth','seed'))
    if type(width) is not int or not 1<=width<=128 or type(depth) is not int or not 1<=depth<=12 or type(seed) is not int:raise ValueError('Invalid benchmark architecture/seed.')
    torch.set_default_dtype(getattr(torch,settings['precision']));torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    points=torch.rand((args.points,2),generator=generator);points[:,0]=2*points[:,0]-1
    initial=torch.column_stack((torch.linspace(-1,1,args.initial),torch.zeros(args.initial)))
    def residual(x,u):return derivative(u,x,1)+u*derivative(u,x,0)-.01*derivative(u,x,0,2)
    problem=PINNProblem(residual,points,[PointConstraint('initial',initial,lambda x:-torch.sin(math.pi*x[:,0:1]))])
    model=DSGNARPINN(problem,PeriodicMLP(width,depth,configuration.get('bias_noise',0.)),weights=dict(pde=.01,initial=1.),
        settings=settings,configuration=configuration).compile(device=args.device)
    model.load_training_state(state);before=model._flat().clone();rng=model.generator.get_state().clone()
    evaluation=np.array([(x,t) for t in np.linspace(0,1,65) for x in np.linspace(-1,1,257)])
    prediction=model.predict(evaluation).detach().cpu().numpy().reshape(-1);reference=burgers_reference(evaluation)
    error=float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))
    assert model.iter==state['iter'] and torch.equal(before,model._flat()) and torch.equal(rng,model.generator.get_state())
    # Check whether an 80-attempt grace would have made any realised stage decision
    # different. This validates equivalence without changing checkpoint metadata.
    replay_settings=dict(model.settings,stage_grace=80);target=replay_settings['target_ratio'];equivalent=True
    for index,row in enumerate(model.diagnostics):
        equivalent=equivalent and row['target_ratio']==target
        target=next_ratio(model.lambda_history[:index+1],replay_settings,target)
    equivalent=equivalent and target==model.target_ratio
    report=dict(scope='Fresh-process checkpoint inference, independently evaluated; no optimizer updates or full protocol/runtime reproduction claim.',
        checkpoint=args.checkpoint.name,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),device=args.device,
        precision=settings['precision'],attempts=model.iter,evaluation_rows=len(evaluation),relative_l2=error,
        reported_single_precision_threshold=4.75e-7,threshold_met=error<=4.75e-7,no_updates=True,
        stage_grace_80_replay_equivalent=equivalent,first_stage2_attempt=next((row['step'] for row in model.diagnostics if row['target_ratio']==settings['stage2_ratio']),None))
    if args.csv:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(('x','t','u','reference','error'))
            writer.writerows((float(p[0]),float(p[1]),float(u),float(r),float(u-r)) for p,u,r in zip(evaluation,prediction,reference))
    text=json.dumps(report,indent=2)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
