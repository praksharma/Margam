"""Official SDK transport/auth adapted from PyniteGUI (AGPL-3.0)."""
import asyncio
import hmac
import ipaddress
import json
from typing import Any

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import StrictInt, ValidationError
from .automation_core import TOOLS


def build_application(bridge, permissions, token, port, network_event, *, require_token=True):
    class PermissionServer(MCPServer):
        async def list_tools(self):
            return [tool for tool in await super().list_tools() if permissions.allows(tool.name)]

        async def call_tool(self, name, arguments, context=None):
            code = ""
            operation_index = None
            if not permissions.allows(name):
                code = "permission_denied" if name in TOOLS else "unknown_tool"
            else:
                try:
                    result = await super().call_tool(name, arguments, context)
                    if result.structured_content and result.structured_content.get("ok") is False:
                        result.is_error = True
                    return result
                except Exception as error:
                    code = "invalid_arguments"
                    cause = error.__cause__
                    if name == "apply_batch" and isinstance(cause, ValidationError):
                        for issue in cause.errors(include_input=False, include_url=False):
                            location = issue["loc"]
                            if len(location) > 1 and location[0] == "operations" and type(location[1]) is int:
                                operation_index = location[1]
                                break
            bridge.logged.emit(f"{name if name in TOOLS else 'unknown_tool'} | {code}")
            response = {"ok": False, "error": {"code": code, "message": "Tool disabled, unknown, or arguments rejected. See its schema and desktop permissions."}}
            if operation_index is not None:
                response["error"]["operation_index"] = operation_index
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(response))],
                                  structured_content=response, is_error=True)

    server = PermissionServer("Margam", log_level="CRITICAL",
        instructions="Controls one open Margam window. Call read_model then read_schema. "
        "Use the returned session_id and exact revision for edits/training. "
        "Only existing numerical parameters can be edited; batches validate before applying. "
        "Training is asynchronous; poll training_status with its job_id. "
        "Outputs require existing explicit output blocks. No arbitrary code or file-management tools.")

    async def call(name, arguments):
        if not permissions.allows(name):
            return {"ok": False, "error": {"code": "permission_denied", "message": "Tool or permission group disabled."}}
        future = bridge.submit(name, arguments)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), 30)
        except TimeoutError:
            bridge.logged.emit(f"{name} | request_timeout")
            return {"ok": False, "error": {"code": "request_timeout", "message": "GUI command timed out; read state before retrying a mutation."}}

    @server.tool(description=TOOLS['read_model'][1], annotations=ToolAnnotations(read_only_hint=True,open_world_hint=False))
    async def read_model() -> dict[str,Any]:
        return await call('read_model',{})

    @server.tool(description=TOOLS['read_schema'][1], annotations=ToolAnnotations(read_only_hint=True,open_world_hint=False))
    async def read_schema(session_id:str) -> dict[str,Any]:
        return await call('read_schema',dict(session_id=session_id))

    @server.tool(description=TOOLS['validate_model'][1], annotations=ToolAnnotations(read_only_hint=True,open_world_hint=False))
    async def validate_model(session_id:str) -> dict[str,Any]:
        return await call('validate_model',dict(session_id=session_id))

    @server.tool(description=TOOLS['apply_batch'][1])
    async def apply_batch(session_id:str,expected_revision:str,operations:list[dict]) -> dict[str,Any]:
        return await call('apply_batch',dict(session_id=session_id,expected_revision=expected_revision,operations=operations))

    @server.tool(description=TOOLS['start_training'][1])
    async def start_training(session_id:str,expected_revision:str) -> dict[str,Any]:
        return await call('start_training',dict(session_id=session_id,expected_revision=expected_revision))

    @server.tool(description=TOOLS['training_status'][1], annotations=ToolAnnotations(read_only_hint=True,open_world_hint=False))
    async def training_status(session_id:str,job_id:str) -> dict[str,Any]:
        return await call('training_status',dict(session_id=session_id,job_id=job_id))

    @server.tool(description=TOOLS['cancel_training'][1])
    async def cancel_training(session_id:str,job_id:str) -> dict[str,Any]:
        return await call('cancel_training',dict(session_id=session_id,job_id=job_id))

    @server.tool(description=TOOLS['read_results'][1], annotations=ToolAnnotations(read_only_hint=True,open_world_hint=False))
    async def read_results(session_id:str,job_id:str,node_id:str|None=None,offset:StrictInt=0,limit:StrictInt=100) -> dict[str,Any]:
        return await call('read_results',dict(session_id=session_id,job_id=job_id,node_id=node_id,offset=offset,limit=limit))

    hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
    origins = [f"http://{host}" for host in hosts]
    app = server.streamable_http_app(stateless_http=True, json_response=True, max_request_body_size=1024*1024,
        transport_security=TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins))

    class LocalAuthentication:
        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await app(scope, receive, send)
            headers = {}
            for key, value in scope.get("headers", []):
                headers.setdefault(key.lower(), []).append(value)
            status, reason = 0, ""
            try:
                local = ipaddress.ip_address(scope["client"][0]).is_loopback
            except (ValueError, TypeError, KeyError):
                local = False
            if not bridge.active:
                status, reason = 503, "server_stopped"
            elif not local or headers.get(b"host") not in ([host.encode()] for host in hosts):
                status, reason = 403, "invalid_host"
            elif b"origin" in headers and headers[b"origin"] not in ([origin.encode()] for origin in origins):
                status, reason = 403, "invalid_origin"
            elif require_token and (len(headers.get(b"authorization", [])) != 1 or not hmac.compare_digest(
                    headers[b"authorization"][0], b"Bearer "+token.encode())):
                status, reason = 401, "unauthorized"
            if status:
                network_event.emit(f"HTTP | {reason}")
                body = json.dumps({"error": reason}).encode()
                response_headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
                if status == 401:
                    response_headers.append((b"www-authenticate", b"Bearer"))
                await send({"type": "http.response.start", "status": status, "headers": response_headers})
                await send({"type": "http.response.body", "body": body})
                return
            network_event.emit("HTTP | authenticated" if require_token else "HTTP | accepted")
            await app(scope, receive, send)

    return LocalAuthentication()
