"""Supplied-point evaluation and comparison exports, independent of the GUI."""
import argparse,json
from dataclasses import asdict
from pathlib import Path


def main():
    import torch
    from margam.templates.examples import example_config
    from margam.templates.poisson import build_pinn
    from margam.components.evaluation_data import read_point_data
    from margam.components.outputs import OutputRequest,evaluate_outputs
    from margam.components.prediction import prediction_model
    from margam.components.export import export_csv
    from margam.components.point_export import export_supplied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=100);parser.add_argument('--output-dir',type=Path,default=Path('build/evaluation-demo'));parser.add_argument('--record',type=Path);args=parser.parse_args()
    if not 1<=args.steps<=100000:parser.error('Use 1–100000 training updates.')
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1];c=example_config('interval_poisson');c.device=args.device;c.iterations=args.steps
    points=read_point_data(dict(path=str(root/'assets/evaluation/supplied-points.csv'),coordinates='x',values='',ids='point_id'),c)
    reference=read_point_data(dict(path=str(root/'assets/evaluation/reference-values.csv'),coordinates='x',values='u',ids='point_id'),c,reference=True)
    model=build_pinn(c);model.train(args.steps,0);before=model.training_state()
    requests=[OutputRequest('comparison','comparison',3,None,points,reference)]
    data=evaluate_outputs(model,c,requests)['comparison'];reopened=prediction_model(c,before);again=evaluate_outputs(reopened,c,requests)['comparison']
    torch.testing.assert_close(torch.tensor(data['values']),torch.tensor(again['values']),rtol=1e-5,atol=1e-7)
    after=model.training_state();unchanged=before['iter']==after['iter'] and before['loss_history']==after['loss_history'] and all(torch.equal(value,after['network'][name]) for name,value in before['network'].items())
    if not unchanged:raise RuntimeError('Output evaluation changed the trained model.')
    args.output_dir.mkdir(parents=True,exist_ok=True);export_csv(args.output_dir/'comparison.csv',data);export_supplied(args.output_dir/'comparison.vtu',data)
    report=dict(scope='Supplied-point functionality and checkpoint inference check, not a paper accuracy benchmark.',device=args.device,torch=str(torch.__version__),configuration=asdict(c),training_steps=model.iter,evaluation_optimizer_updates=0,trained_state_unchanged=unchanged,points=data['points'],point_ids=data['point_ids'],predictions=data['values'],reference=data['reference'],rmse=data['rmse'],max_error=data['max_error'],evaluation=data['evaluation'],reference_provenance=data['reference_provenance'],exports=['comparison.csv','comparison.vtu'])
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.record:args.record.parent.mkdir(parents=True,exist_ok=True);args.record.write_text(text)
    (args.output_dir/'report.json').write_text(text)
    print('Evaluated four supplied rows with exact IDs/order, matched shuffled references and exported CSV/VTU without optimizer updates.')


if __name__=='__main__':main()
