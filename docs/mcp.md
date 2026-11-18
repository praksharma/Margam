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
on an existing connection. Revoking Training prevents new start/resume/cancel requests;
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

## Continue a checkpoint already in the window

With **Training** permission enabled, a client can continue a finished or
cooperatively stopped run, including a run opened through **File → Open saved
run…**. MCP does not choose or open a checkpoint file.

1. Call `read_model` again. Its `continuation` object contains `available`,
   `reason`, completed `steps`, `remaining_updates` and an opaque `run_id`.
   Inspection reads metadata without cloning tensor checkpoint state.
2. When available, call `resume_training` with the latest session, graph revision,
   that `run_id` and an explicit additional count:

   ```json
   {
     "session_id": "FROM_READ_MODEL",
     "expected_revision": "FROM_READ_MODEL",
     "run_id": "FROM_CONTINUATION",
     "additional_steps": 1000
   }
   ```

3. Poll `training_status`, cancel with `cancel_training`, and read losses/outputs
   with `read_results` using the **new `job_id`**. Earlier job IDs are replaced.

The run ID identifies the retained checkpoint separately from the graph. A manual
run can replace weights without changing graph settings; an old run ID is then
rejected. Re-read after any replacement. Run IDs are local session handles, not
portable filenames or archive identifiers.

Continuation uses the same native checkpoint, optimizer/sampler state, application
device, isolated process and optional job time limit as desktop Resume. It changes
the requested total to completed plus additional updates as one normal Undo entry,
then returns the resulting revision. For DSGNAR the count means additional
attempts. The total limit is 100,000; boolean/fractional counts are rejected, as
are unavailable checkpoints and DSGNAR runs stopped at their minimum radius.
The descriptor is an availability hint; the normal execution/data/checkpoint
validators remain authoritative. A worker-side restore failure is reported as a
Failed job, not successful continuation. No file export or view is opened.

A rejected session/revision/run ID, invalid count, busy window, revoked permission
or request cancelled before launch starts no numerical job. Resume consumes
Training permission; it does not require Model edits just to adjust the native
requested total. Undo later restores that total and clears results according to
the normal desktop settings lifecycle.

## Results and limits

Only the latest fresh or resumed job accepted by this MCP session is tracked. Manual training or
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

No headless server, project open/save, checkpoint file loading, data import,
topology editing or file export tools are offered yet. Use the existing desktop
commands for those workflows. The port is for local clients, not an HPC service.

## Validation and source

The original eight dedicated checks cover atomic rejection/Undo, session/revision guards,
queue stop/revocation/limits, real background training and cancellation, result
pagination/staleness, the inverse workflow, optional-dependency isolation and
an official MCP client over authenticated localhost HTTP. The wire test also
checks rejected authentication/origins, live permission changes and token rotation.
These are lifecycle checks, not numerical accuracy benchmarks.

Adapted source and pinned upstream revision are recorded in
[Source attribution](../THIRD_PARTY.md). The protocol implementation is provided
by the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).

Seven additional continuation checks cover missing/stale checkpoint IDs, busy
locks, pending inspector edits, count limits, permission revocation, metadata-only
reads, cancellation, retained and desktop-opened runs, and exact 2+3 versus 5-update
CPU/CUDA weights/history using independent native numerical workers. The official-client test also exercises the new tool,
strict integer schema, new job IDs, paging and live permission revocation.
An earlier parent-process CUDA oracle differed by 2.38e-7 in one PDE loss
reduction; matching thread count alone did not reliably remove it. The qualified
comparison uses independent native workers with the same execution setup, without
relaxing the exact weight/history assertions. This is workflow correctness, not numerical solution accuracy.

## Python tutorial: automate a live Margam window with the official MCP client

When the desktop MCP listener is running (and **Model reads**, **Result reads**,
**Model edits**, and **Training** are enabled in **Automation → MCP server…**),
you can drive the open window from an asynchronous Python script using the
official `mcp` package:

```python
import asyncio
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "http://127.0.0.1:8765/mcp"
MCP_TOKEN = "PASTE_TOKEN_FROM_AUTOMATION_PANEL"

async def call_json(session, tool_name, arguments=None):
    result = await session.call_tool(tool_name, arguments or {})
    payload = json.loads(result.content[0].text)
    if not payload.get("ok"):
        raise RuntimeError(f"{tool_name} failed: {payload.get('error')}")
    return payload["data"]

async def main():
    headers = {"Authorization": f"Bearer {MCP_TOKEN}"}
    async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Read the current open graph and its session_id / revision
            model_info = await call_json(session, "read_model")
            session_id = model_info["session_id"]
            revision = model_info["revision"]
            print("Connected to session:", session_id, "revision:", revision)

            # 2. Discover editable block parameters
            schema = await call_json(session, "read_schema", {"session_id": session_id})
            trainer_node = next(
                b for b in schema["blocks"] if "iterations" in b["editable_parameters"]
            )
            output_node = next(
                (b for b in model_info["nodes"] if b["kind"] in ("solution_plot", "comparison")),
                None,
            )

            # 3. Atomically update trainer iterations and learning rate (one Undo step)
            updated = await call_json(
                session,
                "apply_batch",
                {
                    "session_id": session_id,
                    "expected_revision": revision,
                    "operations": [
                        {"node_id": trainer_node["node_id"], "parameter": "iterations", "value": 200},
                        {"node_id": trainer_node["node_id"], "parameter": "lr", "value": 0.005},
                    ],
                },
            )
            revision = updated["revision"]

            # 4. Validate the graph before launching a background training job
            validation = await call_json(session, "validate_model", {"session_id": session_id})
            print("Model valid:", validation["valid"], "issues:", len(validation["issues"]))

            # 5. Start training and poll until the isolated worker finishes
            job = await call_json(
                session,
                "start_training",
                {"session_id": session_id, "expected_revision": revision},
            )
            job_id = job["job_id"]
            while True:
                status = await call_json(
                    session,
                    "training_status",
                    {"session_id": session_id, "job_id": job_id},
                )
                if not status["running"]:
                    print("Job finished with state:", status["state"], "steps:", status["steps"])
                    break
                await asyncio.sleep(0.25)

            # 6. Read loss history and paginated output predictions
            losses = await call_json(
                session,
                "read_results",
                {"session_id": session_id, "job_id": job_id, "offset": 0, "limit": 10},
            )
            print("Latest loss record:", losses["loss_history"][-1])

            if output_node is not None:
                preds = await call_json(
                    session,
                    "read_results",
                    {
                        "session_id": session_id,
                        "job_id": job_id,
                        "node_id": output_node["id"],
                        "offset": 0,
                        "limit": 5,
                    },
                )
                print("First 5 prediction rows:", preds["output"]["rows"])

if __name__ == "__main__":
    asyncio.run(main())
```
