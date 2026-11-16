# Review memory and workload

Open **Run → Run summary…**, or click **Train PINN** / **F5**. The **Memory and
workload (advisory)** section combines the assembled graph's network, point counts
and output requests. Refresh after edits or when available memory changes.

![A resource review with simulated available RAM](../../assets/screenshots/manual-resources.png)

*Illustration: the available host RAM is explicitly simulated as 32 MiB to show
an advisory warning. Network/workload figures come from the real graph. No
training, memory-exhaustion test or capacity measurement was performed.*

For actual construction/update measurements, use the opt-in
[disposable resource probe](resource-probe.md). It complements this allocation-free
summary and does not predict the full training peak.

## Read the numbers

| Figure | Meaning |
| --- | --- |
| Trainable network parameters | Architecture arithmetic, including optional weight factorization and inverse coefficients; no network is constructed |
| Weight/buffer storage | Scalar storage for the configured network at the displayed default dtype |
| Adam/gradient reference subtotal | Weights, full-size gradients, two Adam moments and network buffers; assumes every parameter has those states, and excludes temporary allocations |
| PDE rows initially / at maximum | Counts include explicitly connected PDE boundary anchors; RAR uses its configured cap and importance sampling keeps its budget |
| Named geometric set rows | Sum of the named sets, distinct from PDE collocation selection |
| Hidden-output scale | Maximum PDE rows × sum of hidden widths × scalar bytes: one hidden-output array per layer, not the saved derivative graph |
| Encoded-input scale | Maximum PDE rows × encoded width × scalar bytes: one raw/feature array; full Fourier encodings can expand rapidly |
| Candidates / scoring chunk | Candidate pool size and the maximum chunk used for residual scoring |
| Prediction rows | Sum across requested output blocks; regular-grid counts are upper bounds for obstacle domains, supplied-cloud counts use exact rows |
| Monitor budgets | Retained record budgets across monitors; the numerical trainer also retains update history |
| Available RAM / VRAM | A point-in-time system/driver query, not a reservation or a fit guarantee |

Several output blocks can request the same locations independently; their rows
are counted separately. Parameter/buffer counts follow the actual supported
architectures. Memory figures use the current default scalar dtype used when a
new desktop network is built. They do not account for an already loaded
checkpoint with a different dtype.

These figures are **not total training or peak memory estimates**. Higher-order
autograd graphs, BC evaluation, optimizer temporaries, gated/multiscale branches,
data copies, SA masks, Scale frozen iterates and Expert NTK/teacher states can
add substantial demand. A small network can still exhaust memory when many
points require PDE derivatives. Histories/outputs also occupy host RAM.

The review builds no network, evaluates no residual and changes neither the
model nor trained results, Undo, samples or RNG state. It reuses Run summary's
existing sample preparation to count rows. A CUDA availability query may
initialize the CUDA context; it does not exercise the training workload.

## Understand an advisory warning

A warning appears when weight/buffer storage, the Adam reference subtotal, the
hidden-output scale or the encoded-input scale individually exceeds **half of the
currently reported available target memory**. This is an advisory review rule,
not an experimentally calibrated prediction or a new hard limit. Train remains
available for valid settings. No samples, architecture, precision or batch
semantics are changed automatically.

Host RAM figures do not include container, scheduler or process quotas. GPU
figures report driver-free memory; PyTorch may also reuse its own allocator cache.
Other programs can change availability immediately after the query. If a query
is unavailable, the review says so and does not invent a capacity estimate.

Existing numerical/configuration bounds still apply, including supported network
sizes, sampler budgets, RAR caps and evaluation-grid limits. A setting passing
those bounds does not establish that it will fit on this machine. Direct API/MCP
training keeps its existing validation; the resource summary is an explicit GUI
review, not a new global allocator policy.

Native prediction forwards are bounded to 2,048 rows, as shown in the review.
Complete outputs are assembled on CPU, so their host memory and transfer costs
still grow with the total requested rows. See [bounded prediction batches](isolated-jobs.md#bounded-prediction-batches)
for scope, measured memory and research API behavior.

## Recover from a memory-allocation failure

Recognized Python/Torch allocation failures in training and prediction produce a
stage-specific message in **Messages**, keeping the backend diagnostic available.

| Stage | Useful action |
| --- | --- |
| Model construction | For a new run, review network width/depth and Fourier feature counts |
| Checkpoint restore | Free memory or choose a suitable device; continuation requires compatible saved architecture/data/objective |
| Training update | For a new run, review active PDE/BC rows, network size, RAR maximum and candidate-scoring chunk |
| Output evaluation | Review output resolution/row count and output requests; training points are independent |

The failed worker releases its model/checkpoint references and produces no valid
new result. Controls become available again after the worker finishes. Previously
saved run files are unchanged; reopen a saved run if you need checkpoint
continuation. A failed optimizer update is not promoted to a valid checkpoint.
Ordinary non-memory errors keep their original diagnostic rather than being
mislabelled as resource failures. The program does not automatically retry a
smaller experiment or change its physics.

Desktop numerical jobs now use [isolated child processes](isolated-jobs.md), with
cooperative Stop and an explicit force-stop/discard action. CPU checkpoint/result
transfer and preparation still need GUI RAM, and system-wide memory pressure or
driver failure can affect the application. OS memory/CPU quotas and general
capacity probes remain planned work; process separation is not a fit guarantee.

## Set an optional job time limit

Open **Run → Job time limit…** and choose the maximum number of seconds for each
numerical job. **No limit (0)** is the default; the desktop control accepts up to
604,800 seconds (seven days). Run summary shows the current choice before training.
The setting applies to new training, checkpoint continuation and output evaluation,
on either CPU or GPU.

![The local time-limit dialog with a five-minute budget](../../assets/screenshots/manual-job-time-limit.png)

*Actual dialog at 300 seconds, captured without running a job or changing the
model. The choice belongs to local desktop preferences.*

The clock covers the numerical process from launch through exit: startup,
construction/restoration, training or inference, and preparing its result. Startup
counts, so a short limit can expire before the first update. Preparing inputs,
validating data or reading returned results in the main window is outside that
clock, as are explicit CSV/ParaView file writing and external geometry editing.

If the limit expires, Margam stops that numerical process and **discards the
running job**. Its partial updates/results are not adopted as a new checkpoint.
Controls become available after the process is reaped, and Messages identifies the
time limit. Previously saved run files remain unchanged; reopen one and choose a
suitable budget before continuing. Normal **Stop** still requests cooperative
cancellation and supported partial-state retention; it must finish before the
configured deadline to return that state.

Changes are locked while a job or external editor is active. The preference is
saved locally across sessions, independently of model/run files. Changing it
preserves model settings, undo history, current results and trained weights.
It limits elapsed calculation time, not memory, samples, precision or update
counts. Direct research scripts keep their explicit execution behavior.

## Pointwise research training chunks

The [research API](../research-microbatch.md) now supports explicit deterministic
pointwise mean-MSE gradient accumulation. The measured 4,096-row FP64 Poisson
case reduces peak CUDA tensor allocation from 36.35 to 17.57 MiB with 256-row
chunks, with exact same-configuration continuation. The first GPU update takes
longer. All points still contribute before a single optimizer update; this does
not automatically reduce samples or change an existing desktop run. Native and
advanced-method training controls remain deferred until their loss/state
contracts are qualified. The case-specific measurement does not replace the
planned representative resource probe for arbitrary native jobs.

## Background references

The implementation is independent. These projects inform the next milestones:

- [Lightning batch-size finding](https://api.lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) measures trial workloads rather than treating parameter counts as sufficient.
- [PhysicsNeMo recommended practices](https://docs.nvidia.com/physicsnemo/26.03/physicsnemo-sym/user_guide/theory/recommended_practices.html) describes gradient aggregation for memory-limited training; Margam does not silently introduce it into existing methods.
- [Ray OOM prevention](https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html) motivates memory monitoring and worker failure containment.
- [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda) distinguishes tensor allocations, allocator reservations and reusable cache.
