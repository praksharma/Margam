"""Short Burgers workflow check; physical setup from the original PINN example."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=100);parser.add_argument('--output',type=Path);parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    if not 2<=args.steps<=100000:parser.error('Use 2–100000 updates.')
    import torch
    from margam.templates.burgers import burgers_config
    from margam.templates.poisson import build_pinn
    from margam.components.outputs import OutputRequest,evaluate_outputs
    from margam.components.paraview_export import export_spacetime_pvd
    torch.set_num_threads(1);config=burgers_config();config.pde_points='both';config.iterations=args.steps
    model=build_pinn(config);split=args.steps//2
    model.train(split,display_every=0);checkpoint=model.training_state()
    resumed=build_pinn(config);resumed.load_training_state(checkpoint)
    model.train(args.steps-split,display_every=0);resumed.train(args.steps-split,display_every=0)
    identical=model.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[key]) for key,value in model.network.state_dict().items())
    if not identical:raise RuntimeError('Burgers continuation changed the training trajectory.')
    data=evaluate_outputs(model,config,[OutputRequest('prediction','paraview',9)])['prediction']
    with TemporaryDirectory() as directory:
        series=Path(directory)/'burgers.pvd';export_spacetime_pvd(series,data)
        entries=ET.parse(series).findall('./Collection/DataSet');times=[float(e.get('timestep')) for e in entries]
        if times!=[i/8 for i in range(9)]:raise RuntimeError('Exported times differ from physical evaluation times.')
        for entry in entries:
            image=ET.parse(series.parent/entry.get('file')).find('ImageData')
            if image.get('WholeExtent')!='0 8 0 0 0 0':raise RuntimeError('Export incorrectly treats time as a spatial axis.')
    if args.export:args.export.parent.mkdir(parents=True,exist_ok=True);export_spacetime_pvd(args.export,data)
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),device='cpu',torch=str(torch.__version__),
                scope='Original problem setup, finite training, exact continuation and physical-time export; no accuracy reproduction.',
                reference='https://maziarraissi.github.io/PINNs/',configuration=asdict(config),coordinates=list(config.coordinate_names),
                constraint_names=list(config.constraint_names),checkpoint_resume_identical=identical,
                initial_losses=model.loss_history[0],final_losses=model.loss_history[-1],evaluation_points=len(data['points']),
                export_frame_times=times,export_spatial_extent='0 8 0 0 0 0',reference_solution_used=False)
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
