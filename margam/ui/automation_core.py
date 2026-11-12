"""Margam GUI-thread commands; queue adapted from PyniteGUI (AGPL-3.0)."""
from concurrent.futures import Future
from dataclasses import dataclass, asdict
from copy import deepcopy
import hashlib
import json
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4
from PySide6.QtCore import QObject, Signal, Slot, Qt

TOOLS = {
    'read_model': ('Model reads', 'Discover this window session, revision and graph. Read before making edits.'),
    'read_schema': ('Model reads', 'Read editable block parameters and their current values. Only existing numerical parameters may be edited.'),
    'validate_model': ('Model reads', 'Run the existing read-only model execution checks and return actionable issues.'),
    'apply_batch': ('Model edits', 'Change existing block numerical parameters atomically as one Undo entry. Requires session_id and expected_revision. No file or topology edits.'),
    'start_training': ('Training', 'Start fresh background training of the current graph. Returns a job ID; poll training_status. Uses application device and existing explicit output blocks.'),
    'training_status': ('Model reads', 'Read the current MCP job status and progress. Check state and error after running becomes false.'),
    'cancel_training': ('Training', 'Request cancellation of the current identified MCP training job; completion may include partial results.'),
    'read_results': ('Result reads', 'Read paginated data from explicit output blocks or retained loss history of the identified finished MCP job. No prediction or file export is triggered.'),
}

EDITABLE = frozenset(('points','sampling','seed','boundary_points','layers','network_type','network_seed','network_options','iterations','lr','pde_weight','boundary_weight','source','viscosity','fourth_order','pde_expression','pde_forcing','bc_values','bc_targets','bc_operators','bc_weights','inverse_settings','gpinn_weights','rba_settings','dsgnar_settings','sa_lr','sa_initial','sa_max','sa_boundary','lra_alpha','lra_interval','lra_epsilon','lra_max'))

class AutomationError(Exception):
    def __init__(self, code, message, operation_index=None):
        self.code, self.message = code, message
        self.operation_index = operation_index
        super().__init__(message)


class Permissions:
    def __init__(self):
        self.lock = Lock()
        self.groups = {group: group.endswith("reads") for group, _ in TOOLS.values()}
        self.tools = {name: True for name in TOOLS}

    def allows(self, name):
        with self.lock:
            return name in TOOLS and self.groups[TOOLS[name][0]] and self.tools[name]

    def set_group(self, name, enabled):
        with self.lock:
            self.groups[name] = bool(enabled)

    def set_tool(self, name, enabled):
        with self.lock:
            self.tools[name] = bool(enabled)


def safe_json(value):
    # Convert numpy scalars and tuples to transportable values; never accept NaN.
    return json.loads(json.dumps(value, allow_nan=False, default=lambda item: item.item()))


class AutomationCommands:
    def __init__(self, window):
        self.window = window
        self.session_id = uuid4().hex
        self.job = None

    def identity(self):
        from .model_document import capture_model
        document = capture_model(self.window)
        # Include the app-wide device, but exclude the mutable inspector config cache.
        content = dict(nodes=document['nodes'], wires=document['wires'], device=self.window.application_device,
                       workspace=id(self.window.project_workspace))
        revision = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return dict(session_id=self.session_id, revision=revision)

    def guard(self, arguments, revision=False):
        identity = self.identity()
        if arguments.get('session_id') != self.session_id:
            raise AutomationError('stale_session', 'Call read_model to discover this window session.')
        if revision and arguments.get('expected_revision') != identity['revision']:
            raise AutomationError('stale_revision', 'The model changed. Read it again before editing or training.')
        return identity

    def status(self):
        job = self.job
        if job is None:
            return dict(**self.identity(), running=self.window.is_running(), state='No MCP job', job_id=None)
        return {**deepcopy(job), **self.identity(), 'job_revision':job['revision'], 'current':job['revision']==self.identity()['revision']}

    def require_job(self, arguments):
        if self.job is None or arguments.get('job_id') != self.job['job_id']:
            raise AutomationError('unknown_job', 'The job was replaced or was not started by this MCP session.')

    def execute(self, name, arguments, cancelled=lambda:False):
        from .model_document import capture_model
        w = self.window
        if name == 'read_model':
            return dict(**self.identity(), model=capture_model(w), busy=w.is_running() or w.scene.locked)
        mutation = name in ('apply_batch','start_training')
        identity = self.guard(arguments, mutation)
        if mutation:
            # Flush a user's pending inspector text before the final revision check.
            # A resulting revision change rejects this command instead of overwriting it.
            w.commit_inspector()
            identity = self.guard(arguments, True)
        if name == 'read_schema':
            from .graph_model import PARAMETERS
            return dict(**identity, blocks=[dict(id=n.identifier, kind=n.kind,
                editable={k:deepcopy(v) for k,v in n.settings.get('parameters',{}).items() if k in EDITABLE and k in PARAMETERS.get(n.kind,())}) for n in w.scene.nodes()],
                batch_example={'operations':[{'node_id':'ID from blocks','parameter':'iterations','value':1000}]},
                rules='Existing parameters only; full numerical validation before commit; file mappings, geometry, topology and method selection are not editable. Read Margam component help for numerical semantics.')
        if name == 'validate_model':
            from .validation_report import check_execution, collect_issues
            error = None
            try: check_execution(w)
            except Exception as caught: error = str(caught)
            return dict(**identity, valid=error is None, execution_error=error,
                        issues=[asdict(issue) for issue in collect_issues(w,error)])
        if name == 'apply_batch':
            if w.is_running() or w.scene.locked or w.project_workspace.busy:
                raise AutomationError('busy', 'Wait for the current job or Jorvik editor to finish.')
            changes = self.candidate(arguments.get('operations'))
            if cancelled(): raise AutomationError('request_cancelled','Command cancelled before applying edits.')
            if not changes: return identity
            def apply(new):
                for node, before, after in changes: node.settings['parameters'] = deepcopy(after if new else before)
                w.config = w.scene.compile_config()
                w.settings_changed()
                w.selection_changed()
            w.push_edit('MCP parameter batch',lambda:apply(True),lambda:apply(False))
            return self.identity()
        if name == 'start_training':
            if w.is_running() or w.scene.locked or w.project_workspace.busy:
                raise AutomationError('busy','Wait for the current job or Jorvik editor to finish.')
            from .validation_report import check_execution
            check_execution(w)
            if cancelled(): raise AutomationError('request_cancelled','Command cancelled before training.')
            if not w.start_training(): raise AutomationError('training_rejected','Current model cannot start training.')
            job = self.job = dict(job_id=uuid4().hex, revision=identity['revision'], running=True,
                state='Running', step=0, error=None)
            worker = w.worker
            self.accepted_worker=worker
            def progress(step, records):
                job['step']=step
            def complete(result):
                job.update(step=result['steps'],state='Partial' if result['cancelled'] else 'Completed')
            def failure(message): job.update(state='Failed',error=message)
            def finished():
                job['running']=False
                if job['state']=='Running': job.update(state='Failed',error='Worker stopped without a completion result.')
            worker.progress.connect(progress);worker.completed.connect(complete)
            worker.failed.connect(failure);worker.finished.connect(finished)
            return self.status()
        if name in ('training_status','cancel_training','read_results'):
            self.require_job(arguments)
            if name == 'training_status': return self.status()
            if name == 'cancel_training':
                if not self.job['running']: raise AutomationError('job_finished','Training has already stopped.')
                if cancelled(): raise AutomationError('request_cancelled','Command cancelled before requesting stop.')
                w.cancel_training();return self.status()
            return self.results(arguments)
        raise AutomationError('unknown_tool','Unknown tool.')

    def candidate(self, operations):
        from .graph_model import PARAMETERS, assemble
        from .model_document import capture_model, decode_document
        if not isinstance(operations,list) or not 1<=len(operations)<=100:
            raise AutomationError('invalid_batch','Supply 1 to 100 parameter changes.')
        if len(json.dumps(operations,allow_nan=False).encode())>128*1024:
            raise AutomationError('invalid_batch','Batch exceeds 128 KiB.')
        document=capture_model(self.window); records={r['id']:r for r in document['nodes']}
        for i,op in enumerate(operations):
            if not isinstance(op,dict) or set(op)!={'node_id','parameter','value'}:
                raise AutomationError('invalid_batch','Use node_id, parameter and value only.',i)
            key=op['parameter']
            record=records.get(op['node_id']) if isinstance(op['node_id'],str) else None
            if not isinstance(op['node_id'],str) or not isinstance(key,str):
                raise AutomationError('invalid_batch','Block IDs and parameter names must be strings.',i)
            if record is None or key not in EDITABLE or key not in PARAMETERS.get(record['kind'],()):
                raise AutomationError('invalid_batch','Unknown block or parameter not editable through MCP.',i)
            record['settings']['parameters'][key]=deepcopy(op['value'])
        # Strict document decoder and the same graph/config validators used by the GUI.
        decode_document(document)
        scene=self.window.scene
        nodes={r['id']:SimpleNamespace(**{k:r[k] for k in ('kind','title')},identifier=r['id'],settings=r['settings'],scene=lambda:scene) for r in records.values()}
        wires=[SimpleNamespace(source=SimpleNamespace(parentItem=lambda n=nodes[r['source']]:n),
            target=SimpleNamespace(parentItem=lambda n=nodes[r['target']]:n,name=r['input'])) for r in document['wires']]
        config=assemble(list(nodes.values()),wires);config.validate()
        changes=[]
        for node in scene.nodes():
            before=node.settings.get('parameters');after=records[node.identifier]['settings'].get('parameters')
            if before!=after: changes.append((node,deepcopy(before),deepcopy(after)))
        return changes

    def results(self, arguments):
        w=self.window;job=self.job
        if job['running']: raise AutomationError('busy','Wait for training to finish before reading its results.')
        if job['revision']!=self.identity()['revision'] or w.last_run is None or w.result is None or job['state'] not in ('Completed','Partial'):
            raise AutomationError('stale_results','Results are unavailable or the graph changed; train again.')
        if w.last_run is not getattr(self,'accepted_worker',w.last_run):
            raise AutomationError('stale_results','Another desktop job replaced these results.')
        offset,limit=arguments.get('offset',0),arguments.get('limit',100)
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=1000:
            raise AutomationError('invalid_query','Use nonnegative offset and limit 1..1000.')
        node_id=arguments.get('node_id')
        if node_id is None:
            rows=w.last_run.model.loss_history
            return dict(**self.status(),kind='loss_history',offset=offset,total_rows=len(rows),rows=deepcopy(rows[offset:offset+limit]))
        data=w.result['outputs'].get(node_id)
        if data is None: raise AutomationError('no_results','Select an explicit evaluated output block ID; omit node_id for loss history.')
        rows=[dict(point=p,value=v) for p,v in zip(data['points'][offset:offset+limit],data['values'][offset:offset+limit])]
        if 'point_ids' in data:
            for row,point_id in zip(rows,data['point_ids'][offset:offset+limit]):row['point_id']=point_id
        return dict(**self.status(),node_id=node_id,fields=data.get('fields',['u']),offset=offset,total_rows=len(data['points']),rows=rows)


@dataclass
class CommandRequest:
    name: str
    arguments: dict
    generation: int
    future: Future


class QtCommandBridge(QObject):
    requested = Signal(object)
    logged = Signal(str)

    def __init__(self, window, permissions):
        super().__init__(window)
        self.commands = AutomationCommands(window)
        self.permissions = permissions
        self.lock = Lock()
        self.active = False
        self.generation = 0
        self.queued = 0
        self.requested.connect(self.dispatch, Qt.ConnectionType.QueuedConnection)

    def set_active(self, active):
        with self.lock:
            self.active = active
            self.generation += 1

    def submit(self, name, arguments):
        future = Future()
        with self.lock:
            if not self.active or self.queued >= 64:
                future.set_result({"ok": False, "error": {"code": "server_unavailable", "message": "Server stopped or command queue full."}})
                return future
            request = CommandRequest(name, arguments, self.generation, future)
            self.queued += 1
        self.requested.emit(request)
        return future

    @Slot(object)
    def dispatch(self, request):
        with self.lock:
            active = self.active and request.generation == self.generation
            self.queued -= 1
        if request.future.cancelled():
            return
        try:
            if not active:
                raise AutomationError("server_stopped", "The server stopped before dispatch.")
            if not self.permissions.allows(request.name):
                raise AutomationError("permission_denied", "This tool or its permission group is disabled.")
            data = safe_json(self.commands.execute(request.name, request.arguments, request.future.cancelled))
            response, status = {"ok": True, "data": data}, "ok"
        except AutomationError as error:
            error_data = {"code": error.code, "message": error.message}
            if error.operation_index is not None:
                error_data["operation_index"] = error.operation_index
            response, status = {"ok": False, "error": error_data}, error.code
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            response, status = {"ok": False, "error": {"code": "invalid_arguments", "message": str(error)}}, "invalid_arguments"
        except Exception:
            response, status = {"ok": False, "error": {"code": "internal_error", "message": "The command failed without a complete response."}}, "internal_error"
        # Fixed names/codes only: no arguments, model values, credentials or paths.
        self.logged.emit(f"{request.name if request.name in TOOLS else 'unknown_tool'} | {status}")
        if not request.future.done():
            request.future.set_result(response)
