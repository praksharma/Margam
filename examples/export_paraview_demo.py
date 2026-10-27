"""Generate a synthetic, known-value playback dataset, not a PINN result."""
import argparse
import math
from pathlib import Path
from margam.components.paraview_export import export_pvd


def demo_frames(count=21,resolution=21):
    axis=[-1+2*i/(resolution-1) for i in range(resolution)]
    points=[[x,y] for y in axis for x in axis]
    for index in range(count):
        time=index/(count-1)
        yield time,dict(points=points,fields=['demo_scalar'],
                        values=[math.sin(math.pi*x)*math.sin(math.pi*y)*math.cos(2*math.pi*time) for x,y in points])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New .pvd path; parent directory must exist')
    args=parser.parse_args()
    if args.output.exists():parser.error('choose a new output filename')
    export_pvd(args.output,demo_frames())
    print('Synthetic playback dataset: '+str(args.output))
    print('Open this PVD in ParaView, Apply, colour by demo_scalar, then Play.')
