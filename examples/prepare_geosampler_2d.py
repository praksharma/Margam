"""Prepare selected planar GeoSampler NPZ datasets without importing CAD libraries."""
import argparse
from margam.components.geosampler_adapter import prepare_planar_export


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--axes',default='x,y',help='Two source axes: x,y or x,z or y,z (order explicit)')
    parser.add_argument('--labels',nargs='+');args=parser.parse_args()
    try:axes=tuple(('x','y','z').index(value.strip()) for value in args.axes.split(','))
    except ValueError:parser.error('Axes must be x, y or z.')
    report=prepare_planar_export(args.input,args.output,axes,args.labels)
    print(f"Prepared {len(report['datasets'])} datasets. Read mapping.json, then map points/normals in Margam. Units/orientation are not inferred.")


if __name__=='__main__':main()
