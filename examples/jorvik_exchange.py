"""Exercise real Jorvik annulus sampling and Margam NetCDF training/resume."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


def sample_annulus(directory):
    from jorvik.geometry import load_geometry
    from jorvik.regions import put_region,region_snapshot
    from jorvik.sampling import sample_geometry
    from jorvik.export import export_samples
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=directory/'annulus.geo'
    source.write_text('SetFactory("OpenCASCADE");\nDisk(1)={0,0,0,1,1};\nDisk(2)={0,0,0,0.5,0.5};\nBooleanDifference{Surface{1}; Delete;}{Surface{2}; Delete;}\n')
    geometry=load_geometry(source);regions=[]
    for label,inside in [('inner',True),('outer',False)]:
        entities=[]
        for entity in geometry['entities']:
            if entity['dimension']!=1:continue
            points=np.asarray(entity['positions']).reshape(-1,3)
            if (np.median(np.linalg.norm(points[:,:2],axis=1))<.75)==inside:entities.append([1,entity['tag']])
        regions=put_region(regions,geometry,label,entities,'#147d92')
    batches=[]
    for region,count in [(regions[0],17),(regions[1],29),(None,97)]:
        request={key:geometry[key] for key in ('source','source_hash','document')}
        request.update(label=region['label'] if region else 'interior',target='region' if region else 'interior',
                       entities=region['entities'] if region else [],method='sobol',seed=42,count=count,
                       resolution=32,compute_mode='reproducible',cpu_threads=1)
        if region:request['boundary_region']=region_snapshot(regions,geometry,region['id'])
        batches.append(sample_geometry(request))
    path=directory/'domain.nc';export_samples(batches,path)
    return path,{'interior':('interior','interior'),'inner':('boundary','inner'),'outer':('boundary','outer')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('build/jorvik-exchange.json'));args=parser.parse_args()
    import torch
    from margam.components.jorvik_data import inspect_selection
    from margam.components.training_data import read_training_sets
    from margam.templates.poisson import PoissonConfig,build_pinn
    with TemporaryDirectory(prefix='margam-jorvik-check-') as directory:
        path,selections=sample_annulus(directory);report=inspect_selection(path,selections)
        lower,upper=report['bounds']
        config=PoissonConfig(shape='Prepared2D',x_min=lower[0],y_min=lower[1],x_max=upper[0],y_max=upper[1],
                             source=0.,layers=(8,8),interior_data=report['interior'],
                             boundary_data={name:item['mapping'] for name,item in report['boundaries'].items()})
        config.bc_types.update(inner='Dirichlet',outer='Dirichlet');config.bc_values.update(inner=1.,outer=0.)
        interior,boundaries=read_training_sets(config)
        for name,sign in [('inner',-1),('outer',1)]:
            points,normals,_=boundaries[name]
            if not np.all(sign*(points*normals).sum(axis=1)>0):raise AssertionError('Annulus outward orientation is wrong.')
        full=build_pinn(config);full.train(4,0)
        part=build_pinn(config);part.train(2,0)
        resumed=build_pinn(config);resumed.load_training_state(part.training_state());resumed.train(2,0)
        assert full.loss_history==resumed.loss_history
        for name,value in full.network.state_dict().items():torch.testing.assert_close(value,resumed.network.state_dict()[name],rtol=0,atol=0)
        result=dict(case='Jorvik annulus / steady scalar Laplace',device='cpu',source_axes=report['axes'],
                    counts=dict(interior=len(interior[0]),**{name:len(rows[0]) for name,rows in boundaries.items()}),
                    outward_normals_verified=True,updates=4,exact_same_device_resume=True,
                    final_loss=full.loss_history[-1],scope='Interchange and training correctness check, not an accuracy benchmark.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
