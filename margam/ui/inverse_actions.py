"""Concrete observation mapping and coefficient reporting for inverse Burgers."""
from copy import deepcopy
from PySide6.QtWidgets import QLabel,QPushButton,QMessageBox
from .setting_help import show_information


class InverseActions:
    def inverse_settings_fields(self):
        self.property_section('Trainable Burgers coefficients')
        for key,label,low,high in [('lambda_1','Initial lambda 1',-100.,100.),('log_lambda_2','Initial log lambda 2',-20.,5.)]:
            from PySide6.QtWidgets import QDoubleSpinBox
            widget=QDoubleSpinBox();widget.setDecimals(6);widget.setKeyboardTracking(False);widget.setRange(low,high);widget.setValue(self.config.inverse_settings[key]);widget.setObjectName('inverse_'+key)
            widget.valueChanged.connect(lambda value,k=key:self.change_inverse_option(k,value));self.form.addRow(label,widget)
        note=QLabel('u_t + lambda_1 u u_x − exp(log_lambda_2) u_xx = 0. The two coefficients train with the network. These are initial guesses, not reference answers. Both losses use the same observed (x,t) rows; no separate BC loss.');note.setWordWrap(True);self.form.addRow(note)
        button=QPushButton('Learned coefficients…');button.clicked.connect(self.show_inverse_parameters);self.form.addRow(button)

    def change_inverse_option(self,key,value):
        values=deepcopy(self.config.inverse_settings);values[key]=value;self.change('inverse_settings',values)

    def edit_observations(self):
        if self.is_running() or self.scene.locked:return
        node=self.parameter_owner('interior_data')
        if node is None or node.kind!='observations':return
        from .data_mapping import DataMappingDialog
        from ..components.inverse_config import read_observations
        config=self.config_for_node(node);source=config.interior_data
        dialog=DataMappingDialog(2,source,self,require_values=True)
        dialog.setWindowTitle('Map training observations (x,t,u)')
        dialog.fields['normals'].clear();dialog.fields['normals'].setEnabled(False)
        if not source:
            dialog.fields['coordinates'].setText('x,t');dialog.fields['values'].setText('u')
        def read(mapped):
            local=deepcopy(config);local.interior_data=mapped;points,values=read_observations(local);return points,None,values
        dialog.read_source=read
        if dialog.exec()!=dialog.DialogCode.Accepted:return
        try:
            mapped=dialog.store_source()
            self.record_dataset_import(node,'interior_data',mapped,lambda:self.change('interior_data',mapped))
        except Exception as error:QMessageBox.warning(self,'Observations could not be mapped',str(error));return
        self.selection_changed()

    def inverse_parameters(self):
        model=self.last_run.model if self.last_run is not None else None
        if model is not None and hasattr(model.network,'estimated_parameters'):
            history=getattr(model,'inverse_history',None)
            if history is None:history=model.training_state().get('inverse',{}).get('history',[])
            return model.network.estimated_parameters(),history
        checkpoint=self.loaded_checkpoint
        if checkpoint and 'inverse' in checkpoint:
            history=checkpoint['inverse']['history']
            if history:return {key:history[-1][key] for key in ('lambda_1','lambda_2')},history
            values=checkpoint['network'];return dict(lambda_1=values['lambda_1'].item(),lambda_2=values['log_lambda_2'].exp().item()),[]
        return {},[]

    def show_inverse_parameters(self):
        values,history=self.inverse_parameters()
        text='No matching inverse run is available. Train or open its run archive first.'
        if values:
            text='Learned Burgers coefficients\n'+ '\n'.join(f'{key}: {value:.9g}' for key,value in values.items())
            text+='\n\nlambda_2 = exp(log_lambda_2), so diffusion remains positive.\nHistory below is after each completed optimizer update; loss curves are before the update.\n\n'
            text+='\n'.join(f'{row["step"]}: lambda_1={row["lambda_1"]:.9g}, lambda_2={row["lambda_2"]:.9g}' for row in history[-200:])
            if len(history)>200:text+='\nShowing the last 200 updates; the checkpoint retains the complete history.'
        return show_information(self,'Inverse Burgers coefficients',text)

    def load_inverse_example(self):
        if self.is_running() or self.scene.locked:return False
        from ..templates.inverse_burgers import inverse_example_config
        from .graph_model import PARAMETERS,parameters_for
        from ..components.inverse_config import read_observations
        try:
            example=inverse_example_config();read_observations(example)
        except (OSError,ValueError) as error:
            QMessageBox.warning(self,'Inverse example unavailable',str(error));return False
        self.reset_template();self._resetting=True
        try:
            self.output_views.sync([]);self.scene.wires.clear();self.scene.clear();self.config=example
            # The bundled fixture is copied; saved/moved models and archives are independent.
            def validate(source):
                local=deepcopy(self.config);local.interior_data=source;return read_observations(local)
            self.config.interior_data=self.project_workspace.store_dataset(self.config.interior_data,copy=True,validator=validate)
            domain=self.make_node('Rectangle','',0,100)
            observations=self.make_node('Observation data','',250,100)
            pde=self.make_node('Inverse Burgers PDE','',510,0)
            network=self.make_node('Neural network','',510,300)
            method=self.make_node('Inverse trainer','',770,100)
            for node in self.scene.nodes():
                if node.kind in PARAMETERS:node.settings['parameters']=parameters_for(node.kind,self.config)
            self.scene.connect_nodes(domain,observations);self.scene.connect_nodes(observations,pde)
            self.scene.connect_nodes(observations,method,'observations');self.scene.connect_nodes(pde,method,'pde');self.scene.connect_nodes(network,method,'network')
            for index,title in enumerate(('Training Monitor','Solution Plot','CSV Export','ParaView Export')):
                node=self.add_node(title);node.setPos(1050,index*140);self.scene.connect_nodes(method,node)
            from .source_fingerprints import accepted_fingerprint
            observations.settings['data_fingerprints']={'interior_data':accepted_fingerprint(self.project_workspace,self.config.interior_data)}
        except (OSError,ValueError) as error:
            self._resetting=False;self.reset_template();QMessageBox.warning(self,'Inverse example could not be prepared',str(error));return False
        finally:self._resetting=False
        self.settings_changed();self.scene.clearSelection();pde.setSelected(True);self.validate_model();self.fit_model()
        self.undo_stack.clear();self.undo_stack.setClean();self.update_document_title()
        self.statusBar().showMessage('Inverse Burgers · 2000 observations · joint field/coefficient training · Adam starting preset')
        return True
