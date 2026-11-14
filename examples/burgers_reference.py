"""Generate explicit original-case Burgers evaluation/reference CSV, without training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from margam.components.burgers_reference import burgers_reference,VISCOSITY


def generate(output,*,spatial_points=257,time_points=11):
    if type(spatial_points) is not int or not 3<=spatial_points<=4097 or spatial_points%2!=1:
        raise ValueError('Choose an odd spatial count from 3 to 4097, including x=0.')
    if type(time_points) is not int or not 2<=time_points<=1001 or spatial_points*time_points>100000:
        raise ValueError('Choose 2–1001 times and at most 100000 total locations.')
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a fresh output folder; reference datasets are not overwritten.')
    x=np.linspace(-1,1,spatial_points);t=np.linspace(0,1,time_points)
    xx,tt=np.meshgrid(x,t);points=np.column_stack((xx.ravel(),tt.ravel()))
    values=burgers_reference(points);check=burgers_reference(points,order=256)
    difference=float(np.max(np.abs(values-check)))
    if difference>1e-10:raise RuntimeError('Reference quadrature did not converge at the selected locations.')
    output.mkdir(parents=True)
    path=output/'reference.csv'
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('point_id','x','t','u'))
        for index,(point,value) in enumerate(zip(points,values[:,0])):
            writer.writerow((f't{index//spatial_points:03d}-x{index%spatial_points:04d}',*point,float(value)))
    root=Path(__file__).resolve().parents[1]
    files=('margam/components/burgers_reference.py','examples/burgers_reference.py')
    metadata=dict(format='margam-burgers-reference',version=1,
        scope='Independent float64 Cole–Hopf reference for the original Burgers setup; evaluation-only, no training or accuracy claim for a PINN.',
        problem=dict(equation='u_t + u*u_x - nu*u_xx = 0',viscosity=VISCOSITY,x_bounds=[-1.,1.],t_bounds=[0.,1.],initial='-sin(pi*x)',walls=0.),
        grid=dict(spatial_points=spatial_points,time_points=time_points,rows=len(points),order='physical time first, x increasing within frame',columns=['point_id','x','t','u']),
        quadrature=dict(order=128,check_order=256,max_absolute_difference=difference,chunk_rows=512),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        numpy=str(np.__version__),csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        references=['https://maziarraissi.github.io/PINNs/','https://people.sc.fsu.edu/~jburkardt/py_src/burgers_exact/burgers_exact.html'],
        reuse='Independent implementation from the Cole–Hopf formula; no upstream solver code or reference arrays copied.')
    (output/'source.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spatial-points',type=int,default=257);parser.add_argument('--time-points',type=int,default=11)
    args=parser.parse_args()
    try:report=generate(args.output,spatial_points=args.spatial_points,time_points=args.time_points)
    except (ValueError,FileExistsError,RuntimeError) as error:parser.error(str(error))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
