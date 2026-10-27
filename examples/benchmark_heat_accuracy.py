"""Small CPU numerical study; no plotting or implicit reference in the trainer."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import math
import torch
from margam.templates.heat import heat_demo_config
from margam.templates.poisson import build_pinn,rectangle_reference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    points=[[i/40,j/40] for j in range(41) for i in range(41)];records=[]
    for method,weight in (('baseline',10.),('baseline',100.),('hard_dirichlet',1.)):
        for seed in (7,17,27):
            config=heat_demo_config();config.seed=seed;config.network_seed=seed;config.boundary_weight=weight;config.method=method
            model=build_pinn(config);model.train(config.iterations,display_every=0)
            prediction=model.predict(torch.tensor(points)).reshape(-1).tolist()
            reference=rectangle_reference(config,points)
            errors=[a-b for a,b in zip(prediction,reference)]
            record={'method':method,'seed':seed,'boundary_weight':weight,'updates':config.iterations,
                    'rmse':math.sqrt(sum(e*e for e in errors)/len(errors)),
                    'max_error':max(map(abs,errors)),'final_loss':model.loss_history[-1]}
            records.append(record);print(json.dumps(record),flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'recorded_utc':datetime.now(timezone.utc).isoformat(),'torch':str(torch.__version__),
                     'device':'cpu','grid':[41,41],'reference':'40 odd sine modes per axis, truncated','records':records},indent=2)+'\n')


if __name__=='__main__':main()
