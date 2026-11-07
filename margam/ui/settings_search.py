"""Search block-owned settings without constructing hidden inspectors or editing graphs."""
import re
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLineEdit,QListWidget,QListWidgetItem,QLabel,QPushButton,QWidget
from .graph_model import PARAMETERS
from .setting_help import HELP
from ..components.network_config import OPTION_SCHEMAS

LABELS={'block_label':'Block label name','lr':'Adam learning rate','iterations':'Training updates epochs','layers':'Hidden layers network widths','network_type':'Network architecture',
 'boundary_weight':'Mean BC boundary loss weight','pde_weight':'PDE loss weight','points':'Interior sampling points','boundary_points':'Boundary sampling points',
 'bc_weights':'Boundary coefficient weight','bc_types':'Boundary condition type','bc_values':'Boundary constant target value',
 'bc_targets':'Boundary target expression','bc_operators':'Boundary custom operator','loss_formulation':'Loss formulation mean integral',
 'sa_boundary':'Adapt boundary constraint weights','pressure_point':'Pressure gauge coordinates','interior_data':'Map imported interior data','boundary_data':'Map imported boundary data','shape':'Domain geometry','method':'PINN method variant','sa_lr':'Mask Adam learning rate','pde_preset':'Predefined PDE',
 'boundary_sampling':'Boundary sampling method count seed','flow_boundary':'Flow boundary conditions','source_data':'Map source data','reference_expression':'Reference expression','resolution':'Evaluation points / axis','history_limit':'Retained updates'}
PREFIXES={'inverse_settings':'inverse','network_options':'network_option','expert_settings':'expert','scale_settings':'scale','rba_settings':'rba','adaptive_sampling':'sampling','gpinn_weights':'gpinn_weight'}
ALIASES={'network_type':'network_architecture','pde_preset':'predefined_pde_choice','pde_forcing':'predefined_forcing'}


def entries(window):
    result=[]
    for node in sorted(window.scene.nodes(),key=lambda n:(n.title,n.identifier)):
        parameters=node.settings.get('parameters',{})
        keys=['block_label']+list(PARAMETERS.get(node.kind,()))
        if node.kind=='annotation':keys+=['annotation_text']
        elif node.kind=='monitor':keys+=['history_limit']
        elif node.kind in ('solution','comparison','export','paraview'):keys+=['resolution']
        elif node.kind=='reference':keys+=['reference_expression']
        elif node.kind in ('evaluation_points','reference_data'):keys+=['source_data']
        for key in keys:
            value=parameters.get(key,node.settings.get(key))
            label=LABELS.get(key,key.replace('_',' ').capitalize())
            if key=='boundary_weight' and node.kind in ('inverse_method','inverse_recipe'):label='Observation data loss weight'
            if key=='interior_data' and node.kind=='observations':label='Map observations (x,t,u)'
            if key=='pressure_point':
                for axis in range(2):result.append((node.identifier,'pressure_point_'+str(axis),'Pressure gauge '+('x' if axis==0 else 'y'),node.title,key))
                continue
            if key=='boundary_data' and isinstance(value,dict):
                for name in value:result.append((node.identifier,'boundary_data.'+name,'Map imported boundary data · '+name,node.title,key))
                continue
            if key=='network_options':
                schema=OPTION_SCHEMAS.get(parameters.get('network_type','fcnn'),{})
                for option,spec in schema.items():result.append((node.identifier,'network_option_'+option,spec['label'],node.title,option))
                continue
            if key=='boundary_sampling':
                try:names=window.config_for_node(node).sides
                except (ValueError,KeyError):names=tuple(value or {})
                for name in names:
                    for option in ('method','points','seed'):result.append((node.identifier,'boundary_sampling.'+name+'.'+option,'Boundary sampling '+option+' · '+name,node.title,option))
                continue
            if key=='flow_boundary' and isinstance(value,dict):
                for side,fields in value.items():
                    for field in fields:
                        for option in ('type','target','operator'):result.append((node.identifier,'flow_'+side+'_'+field+'_'+option,'Flow boundary '+side+' '+field+' '+option,node.title,option))
                continue
            if key=='bc_weights':
                try:names=window.config_for_node(node).constraint_names
                except (ValueError,KeyError):names=tuple(value or {})
                for name in names:result.append((node.identifier,'boundary_coefficient_'+name,'Boundary coefficient weight · '+name,node.title,name))
                continue
            if key in ('bc_types','bc_values','bc_targets','bc_operators') and isinstance(value,dict):
                for name in value:result.append((node.identifier,key+'.'+name,label+' · '+name,node.title,key))
                continue
            if key in PREFIXES and isinstance(value,dict):
                for name in value:
                    if name=='method':continue
                    result.append((node.identifier,PREFIXES[key]+'_'+name,label+' · '+name.replace('_',' '),node.title,name))
                continue
            result.append((node.identifier,ALIASES.get(key,key),label,node.title,key))
    return result


class SettingsSearch(QDialog):
    def __init__(self,window):
        super().__init__(window);self.owner=window;self.setWindowTitle('Find setting');self.resize(680,440)
        layout=QVBoxLayout(self);self.search=QLineEdit();self.search.setPlaceholderText('Learning rate, boundary weight, viscosity…');layout.addWidget(self.search)
        self.results=QListWidget();layout.addWidget(self.results)
        self.status=QLabel('Search owned settings, then select a result to open its block.');self.status.setWordWrap(True);self.status.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(self.status)
        row=QHBoxLayout();layout.addLayout(row)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        select=QPushButton('Go to setting');select.clicked.connect(self.locate);row.addWidget(select)
        close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)
        self.search.textChanged.connect(self.refresh);self.search.returnPressed.connect(self.locate)
        self.results.itemActivated.connect(lambda item:self.locate())

    def refresh(self):
        query=self.search.text().casefold().split();self.results.clear()
        matches=[]
        for record in entries(self.owner):
            node_id,key,label,title,raw=record
            text=(label+' '+title+' '+raw+' '+key.replace('_',' ')+' '+HELP.get(raw,'')).casefold()
            def word_matches(word,value):return bool(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',value)) if len(word)==1 else word in value
            if not all(word_matches(word,text) for word in query):continue
            primary=(label+' '+raw+' '+key.replace('_',' ')).casefold()
            matches.append((-sum(word_matches(word,primary) for word in query),record))
        for _,record in sorted(matches,key=lambda match:(match[0],match[1][2])):
            node_id,key,label,title,raw=record
            item=QListWidgetItem(label+' — '+title+' ['+node_id[:8]+']');item.setData(Qt.ItemDataRole.UserRole,record);self.results.addItem(item)
        if self.results.count():self.results.setCurrentRow(0)
        self.status.setText(f'{self.results.count()} matching setting(s). Refresh after graph edits.')

    def locate(self):
        item=self.results.currentItem()
        if item is None:return
        node_id,key,label,_,raw=item.data(Qt.ItemDataRole.UserRole)
        if not self.owner.locate_block(node_id):self.refresh();self.status.setText('Block no longer exists; results refreshed.');return
        # Rebuild the existing inspector for the requested flow boundary if needed.
        if key.startswith('boundary_coefficient_') and ':' in key:
            side=key.removeprefix('boundary_coefficient_').split(':')[0]
            if side in self.owner.config.sides and self.owner.config.physics=='incompressible_2d':
                self.owner.flow_bc_side=side;self.owner.selection_changed()
        if key.startswith('boundary_data.'):
            self.owner.boundary_side=key.split('.',1)[1];self.owner.selection_changed();key='boundary_data'
        if key.startswith('boundary_sampling.'):
            _,side,option=key.split('.')
            self.owner.boundary_side=side;self.owner.selection_changed()
            key={'method':'boundary_method','points':'boundary_count','seed':'boundary_seed'}[option]
        if key.startswith('flow_'):
            side=key.split('_')[1]
            if side in self.owner.config.sides:
                self.owner.flow_bc_side=side;self.owner.selection_changed()
        controls=self.owner.property_scroll.widget().findChildren(QWidget)
        control=next((w for w in controls if key in (w.property('search_key'),w.property('help_key'),w.objectName())),None)
        if control is None:
            control=next((w for w in controls if str(w.property('setting_label') or '').casefold()==label.casefold()),None)
        if control is None:
            self.status.setText('Opened block. This setting is conditional or inside a compact recipe; expand the recipe or select its active options to expose it.');return
        self.owner.properties_dock.show();self.owner.property_scroll.ensureWidgetVisible(control)
        control.setFocus(Qt.FocusReason.OtherFocusReason)
        self.status.setText('Located '+label+(' (currently unavailable for these settings)' if not control.isEnabled() else ''))
