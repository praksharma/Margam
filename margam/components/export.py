"""Explicit scalar CSV export; separate from views and ParaView integration."""
import csv
import os
from pathlib import Path
import tempfile


def export_csv(path,data):
    path=Path(path);temporary=None
    points,values=data['points'],data['values']
    dimension=len(points[0])
    if dimension not in (1,2) or len(points)!=len(values):raise ValueError('CSV export requires matching scalar point data.')
    try:
        with tempfile.NamedTemporaryFile(mode='w',newline='',encoding='utf-8',dir=path.parent,prefix='.margam-',delete=False) as stream:
            temporary=stream.name;writer=csv.writer(stream)
            writer.writerow([*('x','y')[:dimension],'u'])
            writer.writerows([*point,value] for point,value in zip(points,values))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)
