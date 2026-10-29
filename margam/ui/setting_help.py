"""Expandable setting descriptions; compact inspectors keep their existing controls."""
import re
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QDialogButtonBox,QVBoxLayout,QHBoxLayout,QWidget,QLabel,QToolButton,QTextBrowser,QFormLayout,QComboBox,QAbstractSpinBox,QLineEdit,QCheckBox)

HELP = {
 'iterations':'Number of optimizer updates requested for this run. Full-batch trainers use their complete active collocation set per update. This is an update budget; an epoch over a minibatched dataset is a different quantity. Resume adds updates to the saved iteration count.',
 'lr':'Adam learning rate for network parameters. It controls the step size of gradient descent. Loss-balancing methods change relative loss coefficients independently; SA also has its own ascent rate for adaptive weights.',
 'points':'Requested interior sampling budget. Grid sampling uses complete layers and may return fewer rows. Imported coordinates determine their own count. Boundary rows contribute to the PDE only when their source is connected to its boundary input.',
 'boundary_points':'Default sampling budget per boundary set or edge. Explicit per-set recipes override this default. Initial traces are separate sets in transient examples. Imported data uses the rows stored in its file.',
 'seed':'Seed for point sampling. Fixed recipes repeat their samples; fresh and adaptive trainers derive reproducible update-dependent seeds. Network initialization has a separate seed.',
 'network_seed':'Seed used to initialize the neural network and its random feature mappings. Use a fixed seed when comparing methods; changing it changes the initial function.',
 'layers':'Comma-separated hidden layer widths. Input and output dimensions come from the problem coordinates and predicted fields. For example 32,32 creates two hidden layers with 32 units each.',
 'sampling':'Selects how coordinates are drawn inside the built-in domain. The dropdown descriptions explain each method. These settings create point locations; the arrows choose which losses use them.',
 'source':'Constant used by the selected equation. For the basic Poisson residual the sign convention is laplace(u) + c = 0. A predefined PDE or custom expression defines how this coefficient enters its equation.',
 'viscosity':'Physical viscosity or diffusivity used by the selected residual. Its meaning and required sign depend on the PDE. For unit-speed, unit-length cavity examples the Reynolds number is 1/viscosity.',
 'pde_weight':'Global coefficient multiplying the PDE loss in the training objective. The monitor records raw physical loss separately. Adding boundary collocation changes the rows included in that PDE term.',
 'boundary_weight':'Global boundary-loss coefficient. Named constraints remain separate and are combined according to the trainer. Adaptive methods can additionally adjust their relative weights.',
 'loss_formulation':'Mean MSE averages pointwise squared errors. Integral loss scales uniform-domain and boundary quadrature by their physical measures, summing vector components. Extra boundary PDE rows require a separate measure and are rejected by the current domain-integral mode.',
 'activation':'Smooth nonlinear activation used by the network. PDE losses differentiate the network with respect to coordinates, sometimes to fourth order, so the differentiability of the activation matters.',
 'features':'Number of Fourier or sine features in this architecture. More features increase the encoded input width, parameter count and derivative cost. Fixed random Fourier maps and trainable affine sine maps have different meanings.',
 'scale':'Frequency scale for this Fourier mapping. The architecture determines whether the frequencies are fixed, random or trainable. Larger frequencies increase oscillation and the magnitude of coordinate derivatives.',
 'input_transform':'Semicolon-separated affine expressions applied to physical coordinates before the trainable sine projection, for example x; y; x-1; y-1. Only the restricted affine grammar is accepted. Normalization is an explicit separate option.',
 'normalize_inputs':'Normalize coordinates using configured bounds before applying affine expressions. Leave this off when reproducing mappings expressed in physical coordinates.',
 'skip_connections':'Adds the supported residual/skip paths inside this network architecture. The actual architecture determines where those paths enter.',
 'resolution':'Evaluation resolution for this output block. It is independent of the training samples. Supplied evaluation points preserve their own rows instead of generating a regular grid.',
 'history_limit':'Maximum number of optimizer-update records retained by this monitor. Closing a monitor window leaves recording active while the connected block remains in the graph.',
 'execution_device':'The application Device menu controls training, resumed training and inference for every block. Automatic chooses CUDA when available, otherwise CPU. Data preparation and file writing run on the CPU.',
 'boundary_name':'Identifier for a named boundary set. Physics is assigned downstream. Stable region identities preserve assignments when Jorvik labels change.',
 'adam_learning_rate':'Adam step size for network parameters. SA mask ascent and loss-balancing coefficients use their own controls.',
 'pde_weighted':'Attention-weighted PDE loss is distinct from raw physical PDE residual MSE. RBA applies detached pointwise weights before squaring.',
 'residual':'Restricted mathematical residual expression, equal to zero. Coordinate derivatives are evaluated by automatic differentiation. Collocation arrows determine where the expression is evaluated.',
 'type':'The boundary condition chooses the operator enforced on its points: Dirichlet constrains the value, Neumann the outward normal derivative, and Custom supplies a restricted operator expression.',
 'target':'Desired value of the selected boundary operator. A target expression is evaluated at that boundary’s coordinates. Imported target values require exact row alignment.',
 'operator':'Custom boundary operator expressed with the restricted mathematical grammar. Normal-dependent operators require explicit outward normals on prepared geometry.',
}
HELP.update({
 'hidden_layers':HELP['layers'],
 'network_architecture':'Pointwise neural network mapping physical coordinates to the requested solution fields. Architectures can be exchanged independently of the PINN trainer. Available choices are explained below; their architecture-specific controls appear after selection.',
 'method_variant':'The baseline uses soft weighted PDE and boundary losses. The supported hard-Dirichlet variant embeds compatible constant boundary values in the output transform. Other PINN algorithms have their own recipe and trainer blocks.',
 'pde_expression':'Mathematical residual set equal to zero. Use field names and coordinates with derivatives such as D(u,x), D(u,x,2) and laplace(u). The active problem determines spatial/time axes. Collocation arrows choose the evaluated rows.',
 'frequencies':'Fixed frequencies used by this Fourier encoding, expressed in cycles under the architecture’s coordinate convention. The separate angular Fourier MLP uses angular modes. Affine Fourier mapping exposes its own trainable sine projection.',
 'cycles':'Multiplier applied as 2π times this value to the trainable projection before taking sine. It changes the coordinate oscillation and derivative scale.',
 'first_omega':'SIREN sine frequency multiplier for the first layer, with its frequency-aware initialization.',
 'omega':'SIREN sine frequency multiplier for hidden layers. Higher frequencies amplify coordinate derivatives as well as oscillations.',
 'sampling_interval':'Completed optimizer updates between adaptive sampling events. Interior samples are rescored at the next scheduled event. Connected boundary PDE anchors stay fixed.',
 'sampling_candidates':'Size of the temporary interior candidate pool scored by the current PDE residual. Candidates are distinct from the active training set and from boundary samples.',
 'sampling_chunk':'Maximum candidate rows evaluated together while computing residual scores. This limits derivative-graph memory; it does not change the optimizer batch size.',
 'sampling_batch':'Maximum high-residual interior points added at each RAR event. Duplicate candidate rows are skipped. Fixed boundary PDE anchors remain in the active set.',
 'sampling_maximum':'Total active PDE-row cap for RAR, including connected boundary anchors. It must cover the initial collocation set.',
 'sampling_power':'Exponent applied to residual scores when constructing importance probabilities. Larger values concentrate draws more strongly around large residuals.',
 'sampling_mixture':'Fraction of importance probability assigned by residual scores. The remaining fraction is uniform, keeping every candidate probability positive. Inverse-probability weights correct the sampled interior MSE.',
 'rba_decay':'Retention factor for the previous detached attention value. Each update adds a normalized absolute residual increment. Values near one retain a longer residual history.',
 'rba_rate':'Increment added from the normalized absolute PDE residual. RBA uses the square of the resulting detached attention in its PDE loss; boundary-condition losses remain separate.',
 'rba_initialization':'Zero starts the recurrence with an increment scaled by the attention rate. Normalized starts the first update from the normalized residual itself, then uses the configured rate.',
 'gradient_floor':'Positive denominator floor for gradient/loss balancing. It limits division by very small gradients and is independent of the optimizer learning rate.',
 'expert_chunks':'Physical-time bins for causal PDE weighting. Complete coordinate tuples remain intact. Connected boundary/initial PDE rows enter the bin containing their physical time.',
 'expert_causal_epsilon':'Strength of causal weighting from earlier time-bin losses. Larger values suppress later-bin losses more strongly until earlier bins improve.',
 'expert_balance_every':'Optimizer-update interval between recomputing the selected gradient or NTK-based loss-balancing estimates.',
 'expert_ntk_points':'Maximum sampled rows per constraint set used to estimate an NTK diagonal trace. Higher budgets cost additional parameter-Jacobian evaluations.',
 'expert_momentum':'Fraction of the old loss coefficient retained by the Expert EMA. This convention differs from LRA alpha, which is the fraction of the new estimate.',
 'expert_decay_rate':'Adam learning-rate decay factor over the configured decay interval. It is separate from relative loss balancing.',
 'expert_decay_steps':'Update scale over which the configured Adam decay factor applies.',
 'expert_windows':'Number of sequential physical-time windows. A frozen prediction from the preceding window supplies the next initial condition; each window resets its optimizer and balancing state.',
 'expert_window_steps':'Training update budget per sequential time window. The complete schedule determines the overall update count.',
 'scale_tau_sc':'Scale-PINN coefficient for the sequential current/previous solution correction. The previous optimizer iterate is frozen while computing this update.',
 'scale_tau_alpha':'Scale for the spatial smoothing part of the Scale-PINN correction. Physical time is excluded from the smoothing Laplacian.',
 'scale_gamma':'Spatial smoothing coefficient in the Scale-PINN correction. Some example presets choose it from their physical viscosity.',
 'scale_fresh':'Draw fresh built-in random point sets for each Scale update. PDE collocation follows the connected sources, and the previous network iterate remains frozen for the correction.',
 'scale_strength':'Multiplier for the complete Scale-PINN correction. Zero recovers the ordinary physical PDE objective while preserving the same point set.',
 'factor_mean':'Mean of the initial normally distributed log-scale in random weight factorization. The initial represented matrix is preserved by the factorization.',
 'factor_std':'Standard deviation of the initial log-scale in random weight factorization. Larger values spread the positive row scales more widely.',
})

HELP.update({
 'network_optimiser':'Adam updates network parameters using moving averages of gradients and squared gradients. Trainer loss balancing and SA mask ascent have separate controls. Current desktop trainers use full-batch Adam on their selected active samples.',
 'sa_lr':'Adam ascent rate for self-adaptive pointwise masks. Network parameters still use their own descent learning rate. The squared masks multiply pointwise squared errors.',
 'sa_initial':'Initial positive value of each self-adaptive mask. Its square is the multiplier in the weighted loss. Fixed point identities are required for these masks.',
 'sa_max':'Finite upper bound for self-adaptive mask values. It limits their ascent while preserving nonnegative pointwise weights.',
 'adapt_boundary_constraint_weights':'Enable separate self-adaptive masks for constraint samples as well as PDE samples. The raw physical losses remain available independently of weighted totals.',
 'lra_alpha':'Fraction of the new gradient-ratio estimate used by the LRA moving average. Expert momentum instead describes the retained old fraction.',
 'lra_interval':'Optimizer updates between LRA gradient-ratio balancing events. Adam updates the network on every training step.',
 'lra_epsilon':HELP['gradient_floor'],
 'lra_max':'Finite upper limit on each LRA constraint-loss coefficient, protecting against very small constraint gradients.',
})
for _axis in ('x','y','t'):
    HELP[_axis+'_min']='Lower physical coordinate bound for this problem coordinate. Built-in domains sample within their bounds; prepared-domain bounds support scaling and validation and do not invent geometry or points.'
    HELP[_axis+'_max']='Upper physical coordinate bound for this problem coordinate. It must exceed the corresponding lower bound. Sampling, derivative coordinates and explicit network normalization use the configured physical coordinate convention.'

CHOICES = {
 'tanh':'Smooth hyperbolic tangent.', 'silu':'Smooth sigmoid-weighted linear activation.', 'sin':'Smooth sinusoidal activation.',
 'axis':'Frequencies along individual coordinate axes.', 'full':'Supported combinations of coordinate frequencies.', 'diagonal':'Frequencies acting along diagonal coordinate combinations.', 'gaussian':'Seeded Gaussian random frequencies.',
 'fcnn':'Pointwise fully connected network.', 'fourier_mlp':'Angular Fourier input encoding followed by a shared MLP.',
 'fourier':'Fixed Fourier input features followed by a shared MLP.', 'affine_fourier':'Restricted affine physical inputs followed by a trainable sine projection.',
 'highway_fourier':'Fourier network with a highway gate and carry path.', 'modified_fourier':'Fourier network with multiplicative branch mixing.',
 'multiscale_fourier':'Parallel fixed Gaussian frequency scales.', 'dgm':'Pointwise gated DGM network.', 'siren':'Sine-activated network with frequency-aware initialization.',
 'fourier_filter':'Multiplicative Fourier feature filters.', 'gabor_filter':'Localized multiplicative Gabor filters.',
 'Dirichlet':'Enforces the predicted field value.', 'Neumann':'Enforces its outward normal derivative.', 'Custom':'Enforces a restricted operator expression.',
}


def show_information(parent,title,text):
    dialog=QDialog(parent);dialog.setWindowTitle(title);dialog.setObjectName('information_dialog')
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    layout=QVBoxLayout(dialog);browser=QTextBrowser();browser.setObjectName('information_text');browser.setPlainText(text);layout.addWidget(browser)
    buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);buttons.rejected.connect(dialog.close);layout.addWidget(buttons)
    dialog.resize(650,480);dialog.show();return dialog


def setting_text(label,widget,context=''):
    key=widget.property('help_key') or widget.objectName().removeprefix('network_option_') or re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_')
    text=HELP.get(key) or HELP.get(re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_')) or widget.whatsThis() or widget.toolTip()
    if not text:text=label+' is a setting of the selected block. Its available values and limits are listed below. The block description explains its supported scope.'
    parts=[text]
    if isinstance(widget,QComboBox):
        options=[]
        for index in range(widget.count()):
            value=widget.itemData(index);description=widget.itemData(index,Qt.ItemDataRole.ToolTipRole) or CHOICES.get(value if isinstance(value,str) else widget.itemText(index),'')
            enabled=bool(widget.model().flags(widget.model().index(index,widget.modelColumn(),widget.rootModelIndex())) & Qt.ItemFlag.ItemIsEnabled)
            options.append('• '+widget.itemText(index)+(' (unavailable for this block)' if not enabled else '')+(' — '+description if description else ''))
        parts.extend(['Current selection: '+widget.currentText(),'Available choices:\n'+'\n'.join(options)])
    elif isinstance(widget,QAbstractSpinBox):parts.append(f'Current value: {widget.value()}\nAllowed range: {widget.minimum()} to {widget.maximum()}')
    elif isinstance(widget,QLineEdit):parts.append('Current value: '+widget.text())
    elif isinstance(widget,QLabel):parts.append('Current value: '+widget.text())
    elif isinstance(widget,QCheckBox):parts.append('Current value: '+('enabled' if widget.isChecked() else 'disabled'))
    if context:parts.append('Block description\n'+context)
    return '\n\n'.join(parts)


class HelpFormLayout(QFormLayout):
    def __init__(self,parent,window,context=''):
        super().__init__(parent);self.window=window;self.context=context

    def addRow(self,*args):
        if len(args)==1 and isinstance(args[0],QCheckBox):
            widget=args[0];container=QWidget();row=QHBoxLayout(container);row.setContentsMargins(0,0,0,0);row.addWidget(widget)
            info=QToolButton();info.setObjectName('setting_info_'+re.sub(r'[^a-z0-9]+','_',widget.text().lower()).strip('_'));info.setText('i');info.setAutoRaise(True);info.setAccessibleName('About '+widget.text());info.setToolTip('Explain '+widget.text())
            info.clicked.connect(lambda checked=False:show_information(self.window,widget.text(),setting_text(widget.text(),widget,self.context)));row.addWidget(info);row.addStretch()
            return super().addRow(container)
        if len(args)==2 and isinstance(args[0],str) and isinstance(args[1],QWidget):
            label,widget=args;container=QWidget();row=QHBoxLayout(container);row.setContentsMargins(0,0,0,0)
            caption=QLabel(label);caption.setBuddy(widget);row.addWidget(caption)
            info=QToolButton();info.setText('i');info.setAutoRaise(True);info.setAccessibleName('About '+label);info.setToolTip('Explain '+label)
            info.setObjectName('setting_info_'+re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_'))
            info.clicked.connect(lambda checked=False:show_information(self.window,label,setting_text(label,widget,self.context)))
            row.addWidget(info);row.addStretch();return super().addRow(container,widget)
        return super().addRow(*args)
