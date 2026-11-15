"""Export a small owned-data study, run the real CLI, verify archives/exports.

Two/three updates exercise software correctness, not Burgers shock accuracy.
The producer uses a state-isolated GUI; the CLI/each numerical child are headless.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(device,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a fresh workflow directory.')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import torch
    from PySide6.QtWidgets import QApplication
    from margam.ui.app import Window
    from margam.ui.model_document import capture_model
    from margam.ui.run_document import load_run
    from margam.studies import catalogue
    from margam.study_document import capture,write
    from examples.documentation import validate_public_text
    app=QApplication.instance() or QApplication([]);window=Window(state_directory=False)
    try:
        window.set_application_device('cpu');assert window.load_burgers_reference_example()
        for key,value in [('points',8),('boundary_points',4),('boundary_sampling',{}),('layers',(4,)),('iterations',2)]:window.change(key,value)
        model=capture_model(window);owner=next(row for row in catalogue(model) if row['parameter']=='iterations')
        axes=[dict(node_id=owner['node_id'],parameter='iterations',mode='candidates',text='[2,3]')]
        study,files=capture(model,axes,window.project_workspace,execution=dict(device=device,wall_time_seconds=0),scene=window.scene)
        output.mkdir(parents=True);source=output/'input.margam-study';write(source,study,files)
        result=subprocess.run([sys.executable,'-m','margam.cli','study','run',str(source),'--output',str(output/'run'),'--export'],capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
        report=json.loads((output/'run/summary.json').read_text())
        assert report['headless'] and report['counts']=={'completed':2}
        records=[]
        import csv
        import xml.etree.ElementTree as ET
        for index,steps in enumerate((2,3),1):
            folder=output/f'run/DP{index:04d}';detail=json.loads((folder/'trial.json').read_text());archive=load_run(folder/'artifacts/run.margam-run.pt')
            assert archive['result']['steps']==steps and len(archive['owned_files'])==1
            assert len(detail['exports'])==2
            exported=next(row for row in detail['exports'] if row['kind']=='export');paraview=next(row for row in detail['exports'] if row['kind']=='paraview')
            with (folder/'artifacts'/exported['path']).open() as stream:rows=list(csv.DictReader(stream))
            frames=ET.parse(folder/'artifacts'/paraview['path']).getroot().findall('.//DataSet')
            assert len(rows)==2827 and len(frames)==11
            assert [float(frame.attrib['timestep']) for frame in frames]==sorted({float(row['t']) for row in rows})
            assert all((folder/'artifacts'/frame.attrib['file']).is_file() for frame in frames)
            for artifact in detail['artifacts']:assert hashlib.sha256((folder/'artifacts'/artifact['path']).read_bytes()).hexdigest()==artifact['sha256']
            records.append(dict(trial_id=detail['trial_id'],status=detail['status'],steps=steps,rows=len(rows),frames=len(frames),metrics=detail['metrics'],timing=detail['timing'],artifacts_verified=True))
        evidence=dict(scope='Real CLI serial owned-data study, two/three updates; software lifecycle only, not trained shock accuracy.',
            device=device,study_id=study['study_id'],headless=True,counts=report['counts'],runtime=report['runtime'],
            source_sha256=report['source']['sha256'],workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            study_archive_sha256=report['study_archive_sha256'],records=records)
        text=json.dumps(evidence,indent=2,allow_nan=False)+'\n';validate_public_text(text,'CLI study workflow evidence');(output/'result.json').write_text(text)
        return evidence
    finally:window.undo_stack.setClean();window.close();app.clipboard().clear();app.processEvents()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.device,args.output),indent=2,allow_nan=False))


if __name__=='__main__':main()
