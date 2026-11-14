"""Exercise explicit Burgers reference data, portability, continuation and PVD frames.

Three-plus-two updates check software behavior; they do not solve the shock to
paper accuracy. Reference columns never become training observations.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np


def run(device,output,*,capture=False,overwrite_capture=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a fresh workflow output directory.')
    root=Path(__file__).resolve().parents[1]
    image=root/'assets/screenshots/manual-burgers-reference.png';manifest=image.with_suffix('.json')
    if capture and (image.exists() or manifest.exists()) and not overwrite_capture:raise FileExistsError('Screenshot exists; use --overwrite-capture.')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import torch
    from PySide6.QtWidgets import QApplication,QLabel,QLineEdit,QPlainTextEdit
    from margam.ui.app import Window
    from margam.ui.model_document import capture_model
    from margam.ui.output_views import VectorResultView
    from margam.ui.theme import configure_theme
    from margam.templates.poisson import build_pinn
    from margam.components.error_norms import error_norms
    from examples.documentation import validate_public_text
    torch.set_num_threads(1);app=QApplication.instance() or QApplication([]);configure_theme(app,'light')
    w=Window(state_directory=False);w.resize(1560,1050);w.show()
    def finish():
        deadline=time.monotonic()+45
        while w.worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.003)
        if w.worker is not None or w.result is None:raise RuntimeError(w.log.toPlainText())
    try:
        if not w.set_application_device(device):raise RuntimeError('Selected execution device is unavailable.')
        w.load_example('burgers_1d');expected=w.scene.compile_config()
        if not w.load_burgers_reference_example():raise RuntimeError('Burgers reference example failed to load.')
        if w.scene.compile_config()!=expected:raise RuntimeError('Reference example changed the training problem.')
        sources=[n for n in w.scene.nodes() if n.kind in ('evaluation_points','reference_data')]
        if len(sources)!=2 or not all(n.settings['source_data']['path'].startswith('margam-owned:') for n in sources):raise RuntimeError('Reference sources were not copied into the project.')
        if len({n.settings['source_data']['path'] for n in sources})!=1:raise RuntimeError('Coordinate/reference mappings must share one owned snapshot.')
        w.change('points',16);w.change('boundary_points',6);w.change('boundary_sampling',{});w.change('layers',(8,));w.change('iterations',3)
        if not w.validate_model() or not w.start_training():raise RuntimeError(w.validation.text())
        finish()
        if w.output_views.entries:raise RuntimeError('Output views opened automatically.')
        nodes={n.kind:n for n in w.scene.nodes() if n.kind in ('solution','comparison','export','paraview')}
        data=w.result['outputs'][nodes['comparison'].identifier]
        points=np.asarray(data['points']);reference=np.asarray(data['reference'])
        resource=root/'assets/evaluation/burgers-reference/reference.csv'
        fixture=np.genfromtxt(resource,delimiter=',',names=True,dtype=None,encoding='utf-8')
        if not np.array_equal(points,np.column_stack((fixture['x'],fixture['t']))) or not np.array_equal(reference,fixture['u']):raise RuntimeError('Supplied reference coordinates/values changed.')
        if data['point_ids']!=fixture['point_id'].tolist():raise RuntimeError('Reference IDs changed.')
        if w.result['outputs'][nodes['solution'].identifier]['reference'] is not None:raise RuntimeError('Prediction-only view acquired a hidden reference.')
        if capture:
            w.open_block(nodes['comparison']);view=w.output_views.entries[nodes['comparison'].identifier][1]
            if not isinstance(view,VectorResultView):raise RuntimeError('Expected a transient result table.')
            view.time.setCurrentIndex(view.time.findData(1.))
            w.show_block_properties(next(n for n in sources if n.kind=='reference_data'));w.fit_model();w.statusBar().showMessage('Burgers reference workflow · three updates · comparison data only')
            for _ in range(6):app.processEvents()
            for cls in (QLabel,QLineEdit,QPlainTextEdit):
                for child in w.findChildren(cls):validate_public_text(child.toPlainText() if isinstance(child,QPlainTextEdit) else child.text(),'Burgers reference capture')
            if not w.grab().save(str(image)):raise RuntimeError('Screenshot failed.')
            manifest.write_text(json.dumps(dict(generator='examples.burgers_reference_workflow',scope='Actual comparison at t=1 after three updates; lifecycle only, not an accurate trained shock.',
                device=device,steps=3,rows=len(points),physical_time=1.,file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest()),indent=2)+'\n')
        output.mkdir(parents=True)
        if not w.save_model(output/'model.margam.json') or not w.save_current_run(output/'run.pt'):raise RuntimeError('Portable model/run save failed.')
        if not w.export_block_data(nodes['export'],output/'prediction.csv') or not w.export_block_data(nodes['paraview'],output/'prediction.pvd'):raise RuntimeError('Comparison export failed.')
        rows=list(csv.DictReader((output/'prediction.csv').open()))
        if len(rows)!=len(points) or list(rows[0])!=['source_row','point_id','x','t','u','ref_u','error_u']:raise RuntimeError('CSV schema/row count changed.')
        for index,row in enumerate(rows):
            if int(row['source_row'])!=index or row['point_id']!=data['point_ids'][index] or float(row['ref_u'])!=reference[index] or float(row['error_u'])!=data['values'][index]-reference[index]:raise RuntimeError('CSV reference/error identity mismatch.')
        frames=ET.parse(output/'prediction.pvd').getroot().findall('.//DataSet');times=np.unique(points[:,1])
        if [float(frame.attrib['timestep']) for frame in frames]!=times.tolist():raise RuntimeError('PVD times changed.')
        for frame,physical_time in zip(frames,times):
            piece=ET.parse(output/frame.attrib['file']).getroot().find('.//Piece')
            selected=np.flatnonzero(points[:,1]==physical_time)
            arrays={a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in piece.findall('PointData/DataArray')}
            if int(piece.attrib['NumberOfPoints'])!=len(selected) or not np.array_equal(arrays['ref_u'],reference[selected]) or not np.array_equal(arrays['source_row'],selected):raise RuntimeError('VTU point/reference identity mismatch.')
        report=dict(scope='Explicit Burgers reference workflow, three-plus-two updates; not a trained-accuracy benchmark.',device=device,
            reference_rows=len(points),physical_times=times.tolist(),reference_evaluation_only=True,shared_owned_snapshot=True,
            csv_rows=len(rows),pvd_frames=len(frames),rows_per_frame=int(sum(points[:,1]==times[0])),
            initial_training_steps=3,short_run_error=error_norms(data['values'],data['reference'],fields=['u']),reference_sha256=hashlib.sha256(resource.read_bytes()).hexdigest())
        # Moving the model and reopening the embedded run both use owned data.
        document=capture_model(w);w.undo_stack.setClean()
        if not w.load_model(output/'model.margam.json') or capture_model(w)!=document:raise RuntimeError('Model round trip changed the example.')
        if not w.open_saved_run(output/'run.pt') or not w.resume_training(2):raise RuntimeError('Run continuation failed.')
        finish();continuous=build_pinn(w.config);continuous.train(5,0)
        retained=w.last_run.model.training_state()
        exact=retained['loss_history']==continuous.loss_history and all(torch.equal(value,continuous.network.state_dict()[key].cpu()) for key,value in retained['network'].items())
        if not exact:raise RuntimeError('Burgers reference workflow changed checkpoint continuation.')
        report.update(final_steps=w.result['steps'],exact_resume=True)
        files=('examples/burgers_reference_workflow.py','margam/components/burgers_reference.py','margam/ui/burgers_reference_example.py','margam/templates/burgers.py')
        report['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
        text=json.dumps(report,indent=2,allow_nan=False)+'\n';validate_public_text(text,'Burgers workflow report')
        (output/'result.json').write_text(text)
        return report
    finally:
        if w.worker is not None:
            w.force_stop_training()
            while w.worker is not None:app.processEvents();time.sleep(.003)
        w.undo_stack.setClean();w.close();app.clipboard().clear();app.processEvents()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--capture',action='store_true');parser.add_argument('--overwrite-capture',action='store_true')
    args=parser.parse_args();report=run(args.device,args.output,capture=args.capture,overwrite_capture=args.overwrite_capture)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
