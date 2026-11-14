"""An explicit evaluation-only reference workflow for the fixed Burgers case."""
from pathlib import Path


class BurgersReferenceExample:
    def request_burgers_reference_example(self):
        if not self.is_running() and not self.scene.locked and self.confirm_discard():return self.load_burgers_reference_example()
        return False

    def load_burgers_reference_example(self):
        if self.is_running() or self.scene.locked:return False
        from ..components.evaluation_data import read_point_data
        from ..templates.burgers import burgers_config
        from .source_fingerprints import accepted_fingerprint
        resource=Path(__file__).resolve().parents[2]/'assets/evaluation/burgers-reference/reference.csv'
        source=dict(path=str(resource),coordinates='x,t',values='',ids='point_id')
        try:
            read_point_data(source,burgers_config())
            read_point_data(dict(source,values='u'),burgers_config(),reference=True)
        except (OSError,ValueError) as error:
            self.statusBar().showMessage('Burgers reference fixture unavailable: '+str(error));return False
        if not self.load_example('burgers_1d'):return False
        try:
            # One immutable owned copy supplies both coordinate and field mappings.
            owned=self.project_workspace.store_dataset(source,copy=True,validator=lambda mapping:read_point_data(mapping,self.config))
        except (OSError,ValueError) as error:
            self.statusBar().showMessage('Could not copy Burgers reference data: '+str(error));return False
        self._resetting=True
        points=self.add_node('Evaluation points');points.setPos(740,710)
        reference=self.add_node('Reference data');reference.setPos(740,870)
        for node,mapping in ((points,owned),(reference,dict(owned,values='u'))):
            node.settings['source_data']=mapping
            node.settings['data_fingerprints']={'source_data':accepted_fingerprint(self.project_workspace,mapping)}
        self.rename_block(reference,'Cole–Hopf reference data')
        comparison=self.add_node('Comparison');comparison.setPos(1100,710)
        method=next(n for n in self.scene.nodes() if n.kind=='method')
        self.scene.connect_nodes(method,comparison,'model')
        self.scene.connect_nodes(points,comparison,'points');self.scene.connect_nodes(reference,comparison,'reference')
        solution=next(n for n in self.scene.nodes() if n.kind=='solution');self.scene.connect_nodes(points,solution,'points')
        for output in [n for n in self.scene.nodes() if n.kind in ('export','paraview')]:
            for wire in list(self.scene.wires):
                if wire.target.parentItem() is output:self.scene.wires.remove(wire);self.scene.removeItem(wire)
            output.setPos(1460,540 if output.kind=='export' else 710)
            self.scene.connect_nodes(comparison,output,'comparison')
        note=self.add_node('Note');note.setPos(1100,900)
        self.change_note(note,'Fixed Cole–Hopf reference: nu=0.01/pi, x=[-1,1], t=[0,1], initial -sin(pi*x), zero walls. Evaluation/reference data only. Changing physics does not regenerate this CSV.')
        self._resetting=False;self.scene.clearSelection();reference.setSelected(True)
        self.graph_changed();self.validate_model();self.fit_model()
        self.undo_stack.clear();self.undo_stack.setClean();self.update_document_title()
        self.statusBar().showMessage('Burgers reference check · owned evaluation-only Cole–Hopf samples · no observations in training · views stay closed')
        return True
