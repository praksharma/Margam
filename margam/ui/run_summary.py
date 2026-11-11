"""Read-only execution plan from the supported graph and initial sample builders."""
from copy import deepcopy
import json
from PySide6.QtWidgets import QDialog,QVBoxLayout,QPlainTextEdit,QPushButton,QHBoxLayout,QLabel
from .validation_report import check_execution
from ..components.collocation import boundary_rows
from ..components.network_config import NETWORK_NAMES,network_options
from ..templates.poisson import sample_problem


def execution_plan(window):
    config,_=check_execution(window)
    c=deepcopy(config)
    windows=c.expert_settings['windows'] if c.method=='expert' and c.is_time_dependent else 1
    if windows>1:
        if len(c.bounds[0])==2:c.y_max=c.y_min+(c.y_max-c.y_min)/windows
        else:c.t_max=c.t_min+(c.t_max-c.t_min)/windows
        c.expert_settings['windows']=1
    if c.method=='inverse_burgers':
        from ..components.inverse_config import read_observations
        points,_=read_observations(c);interior=points.tolist();sets={}
    elif c.pde_preset=='kdv_1d':
        from ..templates.kdv import samples
        interior,sets=samples(c,0)
    elif c.is_paper_case:
        from ..templates.expert_examples import samples
        interior,sets=samples(c)
    else:interior,sets=sample_problem(c)
    interior_count=len(interior) if c.pde_points!='boundary' else 0
    boundary_count=len(boundary_rows(c,sets)) if c.pde_points!='interior' else 0
    rows=interior_count+boundary_count
    requests=window.scene.output_requests(config)
    nodes={n.identifier:n for n in window.scene.nodes()}
    text=[f'Device: {config.device} (application selection)',f'PINN method: {config.method}',
          f'Physics: {config.pde_preset or config.physics}','Trainable coefficients: lambda_1, lambda_2=exp(log_lambda_2)' if config.method=='inverse_burgers' else f'Equation coefficients: source c={config.source:g}, viscosity/diffusivity={config.viscosity:g}',f'Coordinates: {", ".join(config.coordinate_names)}',f'Predicted fields: {", ".join(config.field_names)}',
          f'Network: {NETWORK_NAMES[config.network_type]}',f'Hidden layers: {", ".join(map(str,config.layers))}',
          'Network options: '+json.dumps(network_options(config.network_type,config.network_options),ensure_ascii=False),
          f'Optimizer: Adam · learning rate {config.lr:g}',f'Training budget: {config.iterations} updates',
          f'Fixed observation rows · network seed: {config.network_seed if config.network_seed is not None else config.seed}' if config.method=='inverse_burgers' else f'Sampling seed: {config.seed} · network seed: {config.network_seed if config.network_seed is not None else config.seed}',
          f'Loss formulation: {config.loss_formulation}','PDE collocation: observation locations' if config.method=='inverse_burgers' else f'PDE collocation: {config.pde_points}',
          f'Observation/PDE rows: {rows} (same locations); no separate BC/IC loss' if config.method=='inverse_burgers' else f'Initial PDE rows: {rows} = {interior_count} interior + {boundary_count} boundary/initial',
          'Geometric sample sets: '+', '.join(f'{name}: {len(points)}' for name,points in sets.items()),
          'Active loss coefficients (before adaptive balancing): '+', '.join(f'{name}={value:g}' for name,value in config.loss_weights.items()),
          'Requested outputs: '+(', '.join(nodes[r.identifier].title for r in requests) or 'none')]
    if config.method=='inverse_burgers':text+=['Inverse initial guesses: '+json.dumps(config.inverse_settings),'PDE: u_t + lambda_1*u*u_x - exp(log_lambda_2)*u_xx = 0; data and PDE losses use the same observation rows. No separate boundary/initial loss.']
    if config.physics=='incompressible_2d':text+=['Residuals: '+ ' ; '.join(config.flow_residuals)]
    elif config.pde_expression:text+=['Residual: '+config.pde_expression]
    if config.method=='gpinn':text+=['Residual-gradient coefficients: '+json.dumps(config.gpinn_weights)]
    if config.method=='self_adaptive':text +=[f'Mask Adam ascent rate: {config.sa_lr:g}']
    if windows>1:text +=[f'Time marching: {windows} windows × {config.expert_settings["window_steps"]} updates; counts above are for the first window.']
    if config.adaptive_sampling:text+=['Adaptive sampling: '+json.dumps(config.adaptive_sampling)]
    text +=['','Counts describe initial sample rows, not residual components. Duplicate coordinates are preserved. Gauge/mean-velocity anchors are excluded from boundary PDE rows.',
             'Fresh/adaptive sampling can change the active set. Resume restores checkpoint sets. This summary builds no network and performs no training or inference.',
             'Output blocks remain optional. Training does not open their views.']
    from ..components.resources import memory_snapshot,resource_report,report_text
    counts=dict(interior=interior_count,boundary=boundary_count,pde=rows,sets={key:len(value) for key,value in sets.items()})
    resources=resource_report(config,counts,requests,memory=memory_snapshot(config.device),
        monitor_limits=[nodes[r.identifier].settings['history_limit'] for r in requests if r.kind=='monitor'],
        output_names={identifier:node.title for identifier,node in nodes.items()})
    text.append(report_text(config,resources))
    return config,counts,'\n'.join(text)


class RunSummary(QDialog):
    def __init__(self,window):
        super().__init__(window);self.owner=window;self.setWindowTitle('Run summary');self.resize(720,640)
        layout=QVBoxLayout(self)
        note=QLabel('Review the assembled graph. Refresh after edits; Train rechecks the current model.');note.setWordWrap(True);layout.addWidget(note)
        self.text=QPlainTextEdit();self.text.setReadOnly(True);self.text.setObjectName('run_summary_text');layout.addWidget(self.text)
        row=QHBoxLayout();layout.addLayout(row)
        refresh=QPushButton('Refresh');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        self.train=QPushButton('Train new run');self.train.setObjectName('summary_train');self.train.clicked.connect(self.start);row.addWidget(self.train)
        close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)

    def refresh(self):
        if self.owner.is_running() or self.owner.scene.locked:
            self.train.setEnabled(False);self.text.setPlainText('A job or geometry editor is active. Refresh when it finishes.');return False
        try:_,_,text=execution_plan(self.owner)
        except Exception as error:
            self.text.setPlainText('Cannot assemble this model:\n'+str(error)+'\n\nUse Validate for the actionable issue report.');self.train.setEnabled(False);return False
        self.text.setPlainText(text);self.train.setEnabled(True);return True

    def start(self):
        if not self.refresh():return
        if self.owner.start_training():self.close()
