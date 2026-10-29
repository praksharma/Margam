"""Check the prepared triangle workflow with fixed data and optional CSV/VTU exports."""
import argparse,hashlib,json
from dataclasses import asdict
from pathlib import Path


def main():
    import torch
    from margam.templates.prepared_triangle import prepared_triangle_config
    from margam.templates.poisson import build_pinn
    from margam.components.training_data import read_training_sets
    from margam.components.evaluation_data import read_point_data
    from margam.components.outputs import OutputRequest,ReferenceSource,evaluate_outputs
    from margam.components.export import export_csv
    from margam.components.point_export import export_supplied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=100);parser.add_argument('--output-dir',type=Path,default=Path('build/training-data-demo'));parser.add_argument('--record',type=Path);args=parser.parse_args()
    if not 1<=args.steps<=99998:parser.error('Use 1–99998 training updates.')
    root=Path(__file__).resolve().parents[1];directory=root/'assets/training/triangle';torch.set_num_threads(1)
    c=prepared_triangle_config(directory);c.pde_points='both';c.device=args.device;c.iterations=args.steps
    interior,sets=read_training_sets(c);m=build_pinn(c);m.train(args.steps,0)
    resumed=build_pinn(c);resumed.load_training_state(m.training_state());resumed.train(2,0);m.train(2,0)
    identical=m.loss_history==resumed.loss_history and all(torch.equal(value,resumed.network.state_dict()[name]) for name,value in m.network.state_dict().items())
    if not identical:raise RuntimeError('Same-device checkpoint continuation changed the trained state.')
    points=read_point_data(dict(path=str(directory/'evaluation.csv'),coordinates='x,y',values='',ids='point_id'),c)
    ref=ReferenceSource('Manufactured triangle polynomial',lambda rows:[x*x+y*y for x,y in rows])
    data=evaluate_outputs(m,c,[OutputRequest('comparison','comparison',3,ref,points)])['comparison']
    args.output_dir.mkdir(parents=True,exist_ok=True);export_csv(args.output_dir/'triangle.csv',data);export_supplied(args.output_dir/'triangle.vtu',data)
    configuration=asdict(c)
    for source in [configuration['interior_data'],*configuration['boundary_data'].values()]:source['path']=Path(source['path']).relative_to(root).as_posix()
    report=dict(scope='Prepared planar input and checkpoint/export functionality, not a tuned accuracy benchmark.',device=str(m.device),torch=str(torch.__version__),configuration=configuration,training_steps=m.iter,checkpoint_resume_updates=2,exact_same_device_continuation=identical,interior_rows=len(interior[0]),boundary_rows={name:len(rows[0]) for name,rows in sets.items()},evaluation_rows=len(data['points']),rmse=data['rmse'],max_error=data['max_error'],losses=m.loss_history[-1],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.csv')},exports=['triangle.csv','triangle.vtu'])
    text=json.dumps(report,indent=2,allow_nan=False)+'\n';(args.output_dir/'report.json').write_text(text)
    if args.record:args.record.parent.mkdir(parents=True,exist_ok=True);args.record.write_text(text)
    print('Prepared triangle: named mixed boundary sets, exact same-device checkpoint continuation and supplied-point CSV/VTU export passed.')


if __name__=='__main__':main()
