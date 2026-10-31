"""Concrete multi-issue desktop diagnostics; execution validation stays authoritative."""
from copy import deepcopy
from dataclasses import dataclass,replace
import math
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,QPushButton,QTextBrowser
from shiboken6 import isValid
from .graph_model import PARAMETERS,role,overlay
from .ports import INPUTS
from .project_data import collect_sources,inspect_source
from ..components.expressions import Expression
from ..components.network_config import validate_network


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    category: str
    message: str
    node_id: str | None = None
    boundary: str | None = None


def check_execution(window):
    """Run the supported execution gate without changing editor state."""
    config=window.scene.compile_config();config.validate()
    if config.shape=='Prepared2D':
        from ..components.training_data import read_training_sets
        read_training_sets(config)
    requests=window.scene.output_requests(config)
    return config,len(requests)


def collect_issues(window,execution_error=None):
    nodes=window.scene.nodes();issues=[];seen=set()
    incoming={};outgoing={node.identifier:[] for node in nodes}
    def add(category,message,node=None,boundary=None,severity='Error'):
        item=ValidationIssue(severity,category,str(message),node.identifier if node else None,boundary)
        key=(item.node_id,item.category,item.message,item.boundary)
        if key not in seen:issues.append(item);seen.add(key)
    def check(category,node,callback,boundary=None,label=''):
        try:callback()
        except Exception as error:add(category,(label+': ' if label else '')+str(error),node,boundary)
    for wire in window.scene.wires:
        source,target=wire.source.parentItem(),wire.target.parentItem()
        incoming[(target.identifier,wire.target.name)]=source
        outgoing.setdefault(source.identifier,[]).append(target)
        error=window.scene.connection_error(wire.source,wire.target,ignore=wire)
        if error:add('Connections',error,target)
    for kind,label in [('domain','geometry'),('boundary','boundary point source'),('pde','PDE'),('bc','boundary condition'),('method','PINN method')]:
        found=[node for node in nodes if role(node.kind)==kind]
        if len(found)!=1:add('Graph',f'Use exactly one {label} block; found {len(found)}.',found[0] if found else None)
    methods=[node for node in nodes if role(node.kind)=='method']
    used=set()
    def visit(node):
        if node.identifier in used:return
        used.add(node.identifier)
        for (target,_),source in incoming.items():
            if target==node.identifier:visit(source)
    for method in methods:visit(method)
    outputs={'solution','monitor','comparison','export','paraview'}
    active=set(used)
    for node in nodes:
        if node.kind in outputs and any(target==node.identifier for target,_ in incoming):
            visit(node);active.update(used)
    for node in nodes:
        own_in={port:source for (target,port),source in incoming.items() if target==node.identifier}
        core=node.kind in PARAMETERS
        if node.kind in outputs and not own_in:
            add('Outputs','Not connected: this block requests no results.',node,severity='Warning');continue
        if node.kind in outputs:
            required=['model','reference'] if node.kind=='comparison' else [] if 'comparison' in own_in else ['input']
            if 'comparison' in own_in and set(own_in)!={'comparison'}:add('Connections','Use either compared fields OR a PINN/points connection for this export, not both.',node)
        elif core:required=[spec.name for spec in INPUTS[node.kind] if spec.name!='boundary_points']
        else:required=[]
        for port in required:
            if port not in own_in:
                label=next((spec.label for spec in INPUTS[node.kind] if spec.name==port),port)
                add('Connections','Connect '+label+' to this block.',node)
    # Missing inputs leave graph intent ambiguous; do not blame every upstream
    # block for a single broken connection. Check unused cores once ports exist.
    if not any(i.category=='Connections' and i.message.startswith('Connect ') for i in issues):
        for node in nodes:
            if node.kind in PARAMETERS and node.identifier not in used:
                add('Connections','This core block is outside the executable PINN graph. Connect it or remove it.',node)
    # Independent field checks reuse the existing parsers/validators.
    for node in nodes:
        try:config=window.config_for_node(node,create=False)
        except Exception:config=deepcopy(window.config)
        current=role(node.kind)
        own_in={port:source for (target,port),source in incoming.items() if target==node.identifier}
        if node.kind in ('solution','comparison','export','paraview') and own_in and config.shape=='Prepared2D' and 'points' not in own_in and 'comparison' not in own_in:
            add('Output data','Connect Evaluation points to '+node.title+'; a bounding-box grid is not this prepared domain.',node)
        if current=='domain':
            lower,upper=config.bounds
            if not all(math.isfinite(value) for value in (*lower,*upper)) or any(b<=a for a,b in zip(lower,upper)):
                add('Geometry','Each coordinate maximum must exceed its finite minimum.',node)
        if node.kind=='network' or node.kind.endswith('_recipe') or node.kind=='recipe':
            check('Network',node,lambda c=config:validate_network(c.network_type,c.network_options,c.layers))
            if config.network_type=='affine_fourier':
                from ..components.affine_inputs import affine_map
                check('Network',node,lambda c=config:affine_map(c.network_options.get('input_transform',''),c.coordinate_names))
        if current=='pde':
            if config.physics=='scalar' and config.pde_expression:
                check('Physics',node,lambda c=config:Expression(c.pde_expression,len(c.bounds[0]),coordinates=c.coordinate_names))
            if node.kind=='predefined_pde':
                from ..templates.predefined_pdes import validate_predefined
                check('Physics',node,lambda c=config:validate_predefined(c))
            if config.physics=='incompressible_2d':
                for index,text in enumerate(config.flow_residuals):
                    check('Physics',node,lambda text=text,c=config:Expression(text,2,fields=c.field_names),label=('Momentum x','Momentum y','Continuity')[index])
        if current=='bc' and config.physics=='scalar' and not config.is_paper_case and config.pde_preset!='kdv_1d':
            for side in config.sides:
                target=config.bc_targets.get(side,'');operator=config.bc_operators.get(side,'')
                if target:check('Boundary condition',node,lambda text=target,c=config:Expression(text,len(c.bounds[0]),target=True,coordinates=c.coordinate_names),side,label='Target')
                if config.bc_types.get(side)=='Custom':check('Boundary condition',node,lambda text=operator,c=config:Expression(text,len(c.bounds[0]),coordinates=c.coordinate_names),side,label='Custom operator')
                if config.bc_types.get(side) not in ('Dirichlet','Neumann','Custom') or not math.isfinite(config.bc_values.get(side,float('nan'))):add('Boundary condition','Assign a valid type and finite target to '+side+'.',node,side)
        if current=='bc' and config.physics=='incompressible_2d':
            for side in config.sides:
                for field,spec in config.flow_boundary[side].items():
                    if spec['type']!='None':check('Boundary condition',node,lambda text=spec['target'],c=config:Expression(text,2,target=True,fields=c.field_names),side,label=field+' target')
                    if spec['type']=='Custom':check('Boundary condition',node,lambda text=spec['operator'],c=config:Expression(text,2,fields=c.field_names),side,label=field+' operator')
        if current=='method':
            checks={}
            if config.method=='expert':
                from ..components.expert_config import validate_expert_problem
                checks['Expert']=validate_expert_problem
            if config.method=='scale':
                from ..components.scale_config import validate_scale_problem
                checks['Scale']=validate_scale_problem
            if config.method=='gpinn':
                from ..components.gpinn_config import validate_gpinn_problem
                checks['gPINN']=validate_gpinn_problem
            if config.method=='rba':
                from ..components.rba_config import validate_rba_problem
                checks['RBA']=validate_rba_problem
            for validator in checks.values():check('Method compatibility',node,lambda validator=validator,c=config:validator(c))
            if config.loss_formulation=='integral' and config.pde_points!='interior':add('Loss compatibility','Integral loss requires interior quadrature only. Disconnect boundary points from the PDE input.',node)
            if not 0<config.lr<=1:add('Training','Learning rate must be greater than zero and at most one.',node)
            if type(config.iterations) is not int or not 1<=config.iterations<=100000:add('Training','Training updates must be between 1 and 100000.',node)
        if node.kind=='reference' and node.settings.get('reference_expression'):
            texts=node.settings['reference_expression'].split(';')
            if len(texts)!=len(config.field_names):add('Reference','Provide one reference expression per field.',node)
            for text in texts:check('Reference',node,lambda text=text,c=config:Expression(text.strip(),len(c.bounds[0]),target=True,fields=c.field_names,coordinates=c.coordinate_names))
    # All mapped datasets are inspected; unused optional sources are warnings.
    cache={}
    for record in collect_sources(window):
        outcome=inspect_source(record,window.project_workspace,{},cache)
        node=next((node for node in nodes if node.identifier==record['node_id']),None)
        if outcome['error'] or outcome['changed'] or not record['source']:
            add('Input data',record['role']+': '+(outcome['error'] or outcome['status']),node,record['boundary'],severity='Error' if record['node_id'] in active else 'Warning')
        elif record['source'] and record['role']!='Reference' and record['reader']=='points':
            try:
                config=window.config_for_node(node,create=False)
                if config.shape=='Prepared2D' and not any(i.category=='Boundary condition' and i.boundary==record['boundary'] for i in issues):
                    from ..components.training_data import read_training_sets
                    resolved=window.project_workspace.resolve_source(record['source'],create=False)
                    one=replace(config,interior_data={} if record['boundary'] else resolved,boundary_data={record['boundary']:resolved} if record['boundary'] else {})
                    check('Input data',node,lambda one=one:read_training_sets(one),record['boundary'])
            except Exception as error:add('Input data',str(error),node,record['boundary'])
    if execution_error is not None:
        message=str(execution_error)
        if not any(issue.severity=='Error' and (issue.message in message or message in issue.message) for issue in issues):
            # Additional cross-component checks remain authoritative even where no
            # independent checker can safely identify one responsible block.
            if not any(issue.severity=='Error' and issue.category in ('Connections','Graph') for issue in issues):
                lower=message.lower();owner=None
                for tokens,kind in [(('integral','quadrature','hard dirichlet','learning rate','iterations'),'method'),(('dirichlet','neumann','boundary','corner','gauge'),'bc'),(('network','layer','neuron','fourier'),'network'),(('method','loss','learning','iteration','lra','expert','scale','gpinn','rba','adaptive'),'method'),(('residual','pde','expression','forcing'),'pde'),(('coordinate','geometry','domain','bound'),'domain')]:
                    if any(token in lower for token in tokens):owner=next((n for n in nodes if role(n.kind)==kind),None);break
                owner=next((n for n in nodes if n.kind in outputs and n.title in message),owner)
                add('Execution check',message,owner)
    # Invalid optional declarations outside the executed graph are advisory.
    optional={node.identifier for node in nodes if node.kind not in PARAMETERS}
    issues=[ValidationIssue('Warning',i.category,i.message,i.node_id,i.boundary)
            if i.severity=='Error' and i.node_id in optional and i.node_id not in active else i
            for i in issues]
    return issues


class ValidationReportDialog(QDialog):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.issues=[];self.detached=False
        self.setWindowTitle('Model validation');self.setObjectName('validation_report');self.resize(1050,600)
        layout=QVBoxLayout(self);self.summary=QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        note=QLabel('Click an issue to select its block and open settings. Fix the listed issues, then check again. Some checks need valid connections first; additional issues can appear after those are fixed. Validation does not establish solution accuracy.')
        note.setWordWrap(True);layout.addWidget(note)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Severity','Block','Issue'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows);self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.table,2)
        self.details=QTextBrowser();layout.addWidget(self.details,1)
        buttons=QHBoxLayout();self.check_button=QPushButton('Check again');self.select_button=QPushButton('Select block');close=QPushButton('Close')
        buttons.addWidget(self.check_button);buttons.addWidget(self.select_button);buttons.addStretch();buttons.addWidget(close);layout.addLayout(buttons)
        self.check_button.clicked.connect(window.validate_model);self.select_button.clicked.connect(self.select_issue);close.clicked.connect(self.close)
        self.table.itemSelectionChanged.connect(self.show_details);self.table.cellClicked.connect(lambda *args:self.select_issue());self.table.cellActivated.connect(lambda *args:self.select_issue())
        window.undo_stack.indexChanged.connect(self.mark_stale);window.scene.graphChanged.connect(self.mark_stale)

    def detach(self):
        if self.detached:return
        self.detached=True;self.window.undo_stack.indexChanged.disconnect(self.mark_stale);self.window.scene.graphChanged.disconnect(self.mark_stale)

    def mark_stale(self,*args):
        if isValid(self) and isValid(self.window.scene):self.summary.setText('The model changed. Check again to refresh this report.')

    def set_report(self,issues,valid):
        self.issues=list(issues);self.table.setRowCount(len(issues))
        names={node.identifier:node.title for node in self.window.scene.nodes()}
        for index,issue in enumerate(issues):
            for column,value in enumerate((issue.severity,names.get(issue.node_id,'Project'),issue.message)):
                item=QTableWidgetItem(value);item.setToolTip(value);self.table.setItem(index,column,item)
        errors=sum(i.severity=='Error' for i in issues);warnings=len(issues)-errors
        self.summary.setText(('Execution checks passed' if valid else 'Model cannot run yet')+f' · {errors} errors · {warnings} warnings.')
        if issues:self.table.selectRow(0)
        else:self.details.setPlainText('No issues found by the supported execution checks. No training or prediction was performed.')
        self.show_details()

    def current(self):
        row=self.table.currentRow();return self.issues[row] if 0<=row<len(self.issues) else None

    def show_details(self):
        if not isValid(self.window.scene):return
        issue=self.current()
        if issue:self.details.setPlainText(issue.category+'\n'+issue.message+('\nBoundary set: '+issue.boundary if issue.boundary else ''))
        self.select_button.setEnabled(issue is not None and any(node.identifier==issue.node_id for node in self.window.scene.nodes()))

    def select_issue(self):
        issue=self.current()
        if issue is None:return
        node=next((n for n in self.window.scene.nodes() if n.identifier==issue.node_id),None)
        if node is None:self.details.setPlainText('This issue is project-wide, or its block was removed. Check again.');return
        if issue.boundary:
            config=self.window.config_for_node(node)
            if issue.boundary in config.sides:
                self.window.boundary_side=issue.boundary
                if node.kind=='bc' and config.physics=='incompressible_2d':self.window.flow_bc_side=issue.boundary
        self.window.show_block_properties(node);self.window.view.centerOn(node)
