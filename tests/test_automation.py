"""Native command lifecycle and real official-SDK localhost round trips."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from margam.ui.app import Window
from margam.ui.automation_core import AutomationError
from margam.ui.automation_server import dependencies_available
from margam.ui.model_document import capture_model


class AutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.w=Window(state_directory=False);self.w.set_application_device('cpu')
        self.s=self.w.automation_server;self.c=self.s.bridge.commands
        self.identity=self.c.identity()

    def tearDown(self):
        if self.w.is_running():
            self.w.cancel_training();self.wait(lambda:not self.w.is_running())
        self.w.undo_stack.setClean();self.w.close();self.app.clipboard().clear();self.app.processEvents()

    def wait(self,predicate,seconds=15):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.app.processEvents()
            if predicate():return
            time.sleep(.005)
        self.fail('Timed out waiting for desktop/server')

    def args(self,**values):return dict(session_id=self.identity['session_id'],**values)

    def batch(self,operations):
        return self.c.execute('apply_batch',self.args(expected_revision=self.c.identity()['revision'],operations=operations))

    def test_atomic_batch_undo_and_validation(self):
        node=next(n for n in self.w.scene.nodes() if n.kind=='method')
        before=capture_model(self.w);index=self.w.undo_stack.index()
        ops=[dict(node_id=node.identifier,parameter='iterations',value=2),dict(node_id=node.identifier,parameter='lr',value=-1)]
        with self.assertRaises(ValueError):self.batch(ops)
        self.assertEqual(capture_model(self.w),before);self.assertEqual(self.w.undo_stack.index(),index)
        ops[-1]['value']=.002
        changed=self.batch(ops)
        self.assertNotEqual(changed['revision'],self.identity['revision'])
        self.assertEqual(self.w.undo_stack.index(),index+1)
        self.w.undo_edit();self.assertEqual(capture_model(self.w)['nodes'],before['nodes'])
        self.w.redo_edit();self.assertEqual(node.settings['parameters']['iterations'],2)
        with self.assertRaises(AutomationError):
            self.c.execute('apply_batch',self.args(expected_revision=self.identity['revision'],operations=ops))
        with self.assertRaises(AutomationError):
            self.batch([dict(node_id=node.identifier,parameter='interior_data',value={'path':'unrequested.csv'})])
        schema=self.c.execute('read_schema',self.args());self.assertTrue(schema['blocks'])
        before=capture_model(self.w);report=self.c.execute('validate_model',self.args())
        self.assertTrue(report['valid']);self.assertEqual(capture_model(self.w),before)
        revision=self.c.identity()['revision']
        self.w.replace_model(capture_model(self.w))
        self.assertNotEqual(self.c.identity()['revision'],revision)
        current=next(n for n in self.w.scene.nodes() if n.kind=='method')
        with patch.object(self.w,'commit_inspector',side_effect=lambda:self.w.change('iterations',3)):
            with self.assertRaises(AutomationError) as caught:
                self.batch([dict(node_id=current.identifier,parameter='iterations',value=4)])
        self.assertEqual(caught.exception.code,'stale_revision')
        self.assertEqual(current.settings['parameters']['iterations'],3)

    def test_permission_queue_stop_and_revocation(self):
        b=self.s.bridge;b.set_active(True)
        denied=b.submit('apply_batch',{});self.wait(denied.done)
        self.assertEqual(denied.result()['error']['code'],'permission_denied')
        queued=b.submit('read_model',{});b.set_active(False);b.set_active(True)
        self.wait(queued.done);self.assertEqual(queued.result()['error']['code'],'server_stopped')
        pending=b.submit('read_model',{});self.s.permissions.set_group('Model reads',False)
        self.wait(pending.done);self.assertEqual(pending.result()['error']['code'],'permission_denied')
        self.s.permissions.set_group('Model reads',True)
        futures=[b.submit('read_model',{}) for _ in range(65)]
        self.assertEqual(futures[-1].result()['error']['code'],'server_unavailable')
        for f in futures[:-1]:f.cancel()
        self.app.processEvents();self.assertEqual(b.queued,0)

    def test_background_training_results_staleness_and_manual_replacement(self):
        self.w.change('iterations',2);self.w.change('layers',(4,))
        start=self.c.execute('start_training',self.args(expected_revision=self.c.identity()['revision']))
        self.assertTrue(start['running']);job=start['job_id']
        with self.assertRaises(AutomationError):self.c.execute('read_results',self.args(job_id=job))
        self.wait(lambda:not self.w.is_running(),30)
        status=self.c.execute('training_status',self.args(job_id=job));self.assertEqual(status['state'],'Completed')
        rows=self.c.execute('read_results',self.args(job_id=job,offset=1,limit=1))
        self.assertEqual(rows['total_rows'],2);self.assertEqual(len(rows['rows']),1)
        with self.assertRaises(AutomationError):self.c.execute('read_results',self.args(job_id=job,limit=True))
        self.assertTrue(self.w.start_training());self.wait(lambda:not self.w.is_running(),30)
        with self.assertRaises(AutomationError):self.c.execute('read_results',self.args(job_id=job))
        self.w.change('lr',.002)
        self.assertFalse(self.c.execute('training_status',self.args(job_id=job))['current'])

    def test_cancellation_and_session_guards(self):
        self.w.change('iterations',10000);self.w.change('layers',(4,))
        start=self.c.execute('start_training',self.args(expected_revision=self.c.identity()['revision']))
        job=start['job_id'];self.c.execute('cancel_training',self.args(job_id=job))
        self.wait(lambda:not self.w.is_running(),30)
        self.assertEqual(self.c.execute('training_status',self.args(job_id=job))['state'],'Partial')
        with self.assertRaises(AutomationError):self.c.execute('training_status',self.args(job_id='other'))
        with self.assertRaises(AutomationError):self.c.execute('read_schema',dict(session_id='other'))

    def test_inverse_model_uses_existing_observations_and_trainer(self):
        self.w.undo_stack.setClean();self.assertTrue(self.w.load_inverse_example())
        self.w.change('iterations',2);self.w.change('layers',(4,))
        pde=next(n for n in self.w.scene.nodes() if n.kind=='inverse_pde')
        self.batch([dict(node_id=pde.identifier,parameter='inverse_settings',value=dict(lambda_1=.1,log_lambda_2=-6.))])
        start=self.c.execute('start_training',self.args(expected_revision=self.c.identity()['revision']))
        self.wait(lambda:not self.w.is_running(),30)
        self.assertEqual(self.c.execute('training_status',self.args(job_id=start['job_id']))['state'],'Completed')
        values,history=self.w.inverse_parameters();self.assertEqual(len(history),2)
        self.assertNotEqual(values['lambda_1'],.1);self.assertGreater(values['lambda_2'],0)

    def test_optional_dependency_missing_and_panel(self):
        with patch('margam.ui.automation_server.dependencies_available',return_value=False):
            self.assertFalse(self.s.start(0));self.assertEqual(self.s.state,'Unavailable')
        panel=self.w.show_automation_server();self.assertIn('Unavailable',panel.status.text())
        self.assertFalse(self.s.permissions.groups['Training']);self.assertFalse(self.s.permissions.groups['Model edits'])
        panel.close()

    @unittest.skipUnless(dependencies_available(),'Install the optional mcp extra for wire checks')
    def test_official_client_auth_permissions_edits_training_results_and_restart(self):
        self.assertTrue(self.s.start(0));self.wait(lambda:self.s.state=='Running')
        config=self.s.configuration()['mcpServers']['Margam'];self.assertIn('Authorization',config['headers'])
        token=self.s.token;endpoint=self.s.endpoint
        self.s.permissions.set_group('Model edits',True);self.s.permissions.set_group('Training',True)
        node=next(n for n in self.w.scene.nodes() if n.kind=='method')
        network=next(n for n in self.w.scene.nodes() if n.kind=='network')
        solution=self.w.add_node('Solution Plot')
        self.w.scene.connect_nodes(node,solution)
        solution.settings['resolution']=5
        async def run():
            import httpx2
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with httpx2.AsyncClient(trust_env=False) as http:
                for headers in ({},{'Authorization':'Bearer wrong'},{'Authorization':'Bearer '+token,'Origin':'https://untrusted.invalid'}):
                    response=await http.post(endpoint,headers=headers,json={})
                    self.assertEqual(response.status_code,403 if 'Origin' in headers else 401)
            async with httpx2.AsyncClient(headers=config['headers'],trust_env=False) as http:
                async with streamable_http_client(endpoint,http_client=http) as (read,write):
                    async with ClientSession(read,write,read_timeout_seconds=10) as client:
                        await client.initialize()
                        tools={t.name for t in (await client.list_tools()).tools}
                        self.assertIn('start_training',tools)
                        response=await client.call_tool('read_model',{});self.assertFalse(response.is_error,response)
                        data=response.structured_content['data'];identity={'session_id':data['session_id']}
                        edit=await client.call_tool('apply_batch',{**identity,'expected_revision':data['revision'],'operations':[
                            {'node_id':node.identifier,'parameter':'iterations','value':2},
                            {'node_id':network.identifier,'parameter':'layers','value':[4]}]})
                        self.assertFalse(edit.is_error,edit);revision=edit.structured_content['data']['revision']
                        start=await client.call_tool('start_training',{**identity,'expected_revision':revision});self.assertFalse(start.is_error,start)
                        job=start.structured_content['data']['job_id']
                        for _ in range(500):
                            status=(await client.call_tool('training_status',{**identity,'job_id':job})).structured_content['data']
                            if not status['running']:break
                            await asyncio.sleep(.01)
                        self.assertEqual(status['state'],'Completed',status)
                        result=await client.call_tool('read_results',{**identity,'job_id':job,'offset':0,'limit':1})
                        self.assertFalse(result.is_error,result);self.assertEqual(result.structured_content['data']['total_rows'],2)
                        output=await client.call_tool('read_results',{**identity,'job_id':job,'node_id':solution.identifier,'offset':1,'limit':2})
                        self.assertFalse(output.is_error,output)
                        page=output.structured_content['data'];self.assertEqual(page['total_rows'],25);self.assertEqual(len(page['rows']),2)
                        self.assertEqual(page['fields'],['u'])
                        # Permission revocation affects both discovery and calls on the existing connection.
                        self.s.permissions.set_group('Training',False)
                        self.assertNotIn('start_training',{t.name for t in (await client.list_tools()).tools})
                        denied=await client.call_tool('start_training',{**identity,'expected_revision':revision})
                        self.assertTrue(denied.is_error)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(asyncio.run,run());self.wait(future.done,45);future.result()
        self.s.stop();self.wait(lambda:self.s.thread is None)
        self.assertFalse(self.s.token);self.assertTrue(self.s.start(0));self.wait(lambda:self.s.state=='Running')
        self.assertNotEqual(self.s.token,token)

    def test_desktop_import_does_not_load_optional_server_dependencies(self):
        code="""
import sys
from margam.ui.app import Window
from PySide6.QtWidgets import QApplication
app=QApplication([]);w=Window(state_directory=False)
assert not any(name.split('.')[0] in ('mcp','mcp_types','uvicorn','starlette','pydantic') for name in sys.modules)
w.close();app.processEvents()
"""
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',QT_QPA_PLATFORM='offscreen'))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
