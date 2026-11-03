# Control the open application with MCP

Margam offers an optional MCP server for an AI client to inspect and train the
model in the open desktop window. It uses the official Python MCP SDK and the
localhost transport, native permissions panel and Qt command queue adapted from
[PyniteGUI](https://github.com/praksharma/PyniteGUI).

## Install and connect

Use the existing root environment:

```sh
uv pip install --python .venv/bin/python -e '.[mcp]'
.venv/bin/python -m margam.ui
```

Open **Automation → MCP server…**, choose an unused port and press **Start**.
The default is 8765. Each window needs its own port. The server starts off on
every launch; closing the panel leaves it running. Stop it from this panel or
close the application to disconnect clients.

![Margam MCP permissions panel](../assets/screenshots/manual-mcp.png)

Press **Copy configuration** and use the JSON in a client supporting Streamable
HTTP and bearer headers. Client setup locations differ; the copied configuration
has this shape:

```json
{
  "mcpServers": {
    "Margam": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {"Authorization": "Bearer TOKEN_FROM_THE_PANEL"}
    }
  }
}
```

Tokens change on every start. Keep the real configuration private. A screenshot
or public issue should omit the token. Margam does not register itself in your
client or change client configuration files.

## Choose what the client may do

**Model reads** and **Result reads** start enabled. **Model edits** and
**Training** start disabled. A tool requires both its group and individual
checkbox. Permission changes apply to discovery and each queued command, even
on an existing connection. Revoking Training prevents new start/cancel requests;
it does not stop a job already accepted. Use the native **Stop** button to stop it.

This first version keeps manual editing available while idle. The GUI thread
serializes accepted commands; edits and training starts require the exact model
revision the client read. Training and Jorvik retain their usual busy locks.
Unlike PyniteGUI, Margam does not yet have an exclusive MCP editing mode.

## Client workflow

1. Call `read_model` without arguments. Reuse its `session_id` and `revision`.
   A session identifies this window; graph replacements change the revision.
2. Call `read_schema` with that session ID. It lists each existing block's ID,
   current editable numerical parameters and a batch example. Use these exact
   names; numerical meanings are documented in [Components](components.md).
3. Call `validate_model` to receive execution validity and block-linked issues.
   Inspection does not change settings, files or training results.
4. If Model edits is enabled, use `apply_batch` with `session_id`,
   `expected_revision` and `operations`. For example, replacing the placeholder ID:

   ```json
   {
     "session_id": "FROM_READ_MODEL",
     "expected_revision": "FROM_READ_MODEL",
     "operations": [
       {"node_id": "TRAINER_ID", "parameter": "iterations", "value": 1000},
       {"node_id": "TRAINER_ID", "parameter": "lr", "value": 0.001}
     ]
   }
   ```

   Every candidate parameter and the assembled numerical configuration validate
   before committing. A rejected batch applies nothing. An accepted batch is one
   native Undo step and clears obsolete results using the normal editor lifecycle.
   Batches contain 1–100 operations and at most 128 KiB. Only existing numerical
   parameters are editable: no block insertion, geometry changes, wiring, file
   mappings, arbitrary code, file management or method replacement.
5. With Training enabled, call `start_training` using the latest revision. This
   starts a **fresh** run using the application's device and returns `job_id`
   immediately. It does not open the desktop run-summary dialog. Poll
   `training_status` using `session_id` and `job_id` until `running` is false.
   Check `state` and `error`: Completed, Partial after cancellation, or Failed.
6. `cancel_training` requests a stop for that exact job. Cancellation is checked
   between existing training batches; it does not terminate an optimizer update.
7. Read retained losses with `read_results`, supplying the session and job IDs.
   Add `node_id` for prediction rows from an existing connected output block.
   Use `offset` and `limit` (1–1000). Returned points, values and supplied point IDs
   preserve output row order. No new prediction or export is triggered.

Success returns `{ "ok": true, "data": ... }`; tool failures use
`{ "ok": false, "error": { "code": ..., "message": ... } }` and MCP's
error flag. A stale revision requires a fresh read, not a blind retry. A timed-out
mutation may already have completed: read state before retrying it.

## Results and limits

Only the latest job started by this MCP session is tracked. Manual training or
evaluation can replace its results; changed graphs invalidate result reads.
Status distinguishes graph currency from successful completion. Loss rows are
native numerical history, not reconstructed monitor curves. Explicit output
blocks remain required for predicted data; MCP does not silently add views,
references, evaluation points or export files. Training supports the same graph
and method combinations as the desktop, including inverse Burgers observations.

The listener binds only to IPv4 loopback and requires a fresh bearer token.
Host and Origin checks reject nonlocal/browser origins. Request bodies are
bounded to 1 MiB; at most 64 commands wait for the GUI thread. Logs contain tool
names/status codes, without tokens, arguments, model values or paths. Model reads
intentionally expose the open graph, including its configured input references,
to authorized clients. Optional server packages are loaded only when Start is
pressed; ordinary desktop startup does not import them.

No headless server, project open/save, checkpoint continuation, data import,
topology editing or file export tools are offered yet. Use the existing desktop
commands for those workflows. The port is for local clients, not an HPC service.

## Validation and source

Eight dedicated checks cover atomic rejection/Undo, session/revision guards,
queue stop/revocation/limits, real background training and cancellation, result
pagination/staleness, the inverse workflow, optional-dependency isolation and
an official MCP client over authenticated localhost HTTP. The wire test also
checks rejected authentication/origins, live permission changes and token rotation.
These are lifecycle checks, not numerical accuracy benchmarks.

Adapted source and pinned upstream revision are recorded in
[Source attribution](../THIRD_PARTY.md). The protocol implementation is provided
by the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
