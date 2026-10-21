# FastH3 Omni service package

This package targets FastVideo commit `cdda1b7535fe32699335e39db82c1d4db21ceb88` on
the rebuilt `popos-5090` host. The default root is `/home/isaac/workdir/h3`, the
environment is `envs/omni`, and the upstream checkout is `src/FastVideo`.
It does not use the old H3 conda environments or the previous absent project root.

## Configuration

`configs/fasth3-omni-sp1.template.json` and `configs/fasth3-omni-sp2.template.json`
are complete nested FastVideo serve configurations. FastVideo accepts JSON
natively. The controller replaces exact `${VARIABLE}` values before validation;
do not pass an unrendered template directly to FastVideo.

Both retain NVFP4 QAD, FP4 VSA with the FP8 prefix, two-level P and K smoothing,
TE offload, resident DiT, sequential denoise loading, VAE tiling, and eager DiT.
SP1 uses the official MP executor, GPU1 by default. SP2 uses Ray, GPUs 0,1,
TP1/SP2 and parallel VAE encode/decode, with DiT FSDP disabled. Model weights are
replicated on the two GPUs. Sampling defaults to 832×480, 362 frames, 24 fps,
8 fused forwards, CFG1. The model contract determines the PDD grid and shifts.
Clients can override server defaults; the benchmark client must enforce its
declared sampling contract.

Set these verified absolute asset paths before `render`, `validate` or `start`:

```bash
export H3_OMNI_ROOT=/home/isaac/workdir/h3
export OMNI_MODEL_DIR=/absolute/path/to/complete-or-composed-QAD-model
export OMNI_ADALN_TABLE=/absolute/path/to/matching-QAD-adaln-table.pt
export OMNI_DECODER=/absolute/path/to/36L/decoder-nvfp4.safetensors
# Recommended: immutable manifest recording all component revisions and hashes.
export OMNI_ASSET_MANIFEST=/absolute/path/to/verified-assets.json
python3 omni-deployment/scripts/omni_control.py render sp1
python3 omni-deployment/scripts/omni_control.py validate sp1
python3 omni-deployment/scripts/omni_control.py start sp1
python3 omni-deployment/scripts/omni_control.py status
```

`OMNI_ASSET_MANIFEST` is optional; the controller fingerprints it when supplied.
Without it, the controller hashes the sampling contract, AdaLN table and decoder,
and checks component presence. That does not independently verify every model
shard: use immutable model directories and the asset preparation manifest.

Optional overrides: `OMNI_FASTVIDEO_SRC`, `OMNI_PYTHON`, `OMNI_INPUT_DIR`,
`OMNI_OUTPUT_DIR`, `OMNI_PORT` (8108), `OMNI_GPU_IDS`, `OMNI_MEMO_ENTRIES` (16),
`OMNI_RAY_CPUS` (8), `OMNI_RAY_OBJECT_STORE_BYTES` (4294967296 = 4 GiB).
`OMNI_CUDA_HOME` and `OMNI_MEDIA_BIN` override the CUDA toolkit and media tool
directories with absolute paths. By default the controller derives CUDA from
the unresolved venv interpreter path at
`<venv>/lib/python3.12/site-packages/nvidia/cu13`, sets `CUDA_HOME` and
`CUDACXX=<CUDA_HOME>/bin/nvcc`, and prefixes child `PATH` with venv `bin`, CUDA
`bin`, and `<root>/.tools/media/bin` before the inherited `PATH`. This applies to
both validation and startup, so a fresh shell can start a service that finds
NVCC for JIT compilation and the project's ffmpeg/ffprobe. It preserves uv's
venv path instead of resolving its Python symlink to the managed interpreter.
The service binds only `127.0.0.1`; only files within `OMNI_INPUT_DIR` may be
submitted as local references (default `/home/isaac/workdir/h3/inputs`). The
source checkout HEAD must match the pin.
Inherited `FASTVIDEO_*`, `NCCL_*`, and `RAY_*` overrides are removed before
applying this recipe so another experiment cannot silently change its backend.

`OMNI_NVFP4_MM_BACKEND` explicitly selects `auto` (default), `cutlass`, or
`cudnn` for the pinned NVFP4 DiT matrix multiply path. `OMNI_STAGE_LOGGING`
accepts exactly `0` or `1` and defaults to `1`. Both effective values enter the
service fingerprint and require a fresh service when changed. Row splitting
remains explicitly `FASTVIDEO_NVFP4_MM_ROW_SPLIT=off`; inherited backend,
row-split and logging settings cannot override the controller recipe. These
options expose experiments, not a measured speedup or guaranteed bitwise
equivalence between kernels. Backend support still needs a real target smoke.

All ranks emit stage logs (`FASTVIDEO_LOG_ALL_PROCESSES=1`); retain private Ray
actor logs to map each worker UUID to its GPU. `OMNI_SP_PROFILE=1` enables a
separate diagnostic run: count rank-0 `H3_STAGE_MS` lines within each denoising
stage to measure completed transformer forwards. It adds CUDA events and
synchronization, so the default is `0` and formal timing runs keep it disabled.
Changing the flag changes the service fingerprint and requires stop/start.
Stage timings are seconds in structured results and milliseconds in text logs.
With `OMNI_STAGE_LOGGING=0`, ordinary stage timings and their peak-reset behavior
are disabled; external resource monitoring remains necessary. Compare timings
only against a baseline using the same logging setting.
The executor's final `peak_memory_mb` is not the whole-request peak because
stage logging resets allocator peaks; use the maximum stage peak per rank and
external GPU/host samples. Report the latter as device-level sampled memory.

Two optional memory settings accept exactly `0` or `1`; both default to `0`:

- `OMNI_VAE_OFFLOAD=1` sets `generator.engine.offload.vae=true`. The pinned
  native offload path moves both video and audio VAEs off the GPU between
  relevant stages and clears video NVFP4 decoder graphs when moving to CPU.
  It keeps the same model weights, decoder, reference limit and sampling
  settings, but adds PCIe transfers and graph rebuilding. It does not guarantee
  that every reference/frame combination will fit.
- `OMNI_EXPANDABLE_SEGMENTS=1` sets
  `PYTORCH_ALLOC_CONF=expandable_segments:True`. At `0`, the controller explicitly
  sets `PYTORCH_ALLOC_CONF=backend:native`. Both validation and startup remove
  inherited `PYTORCH_ALLOC_CONF` and legacy `PYTORCH_CUDA_ALLOC_CONF` before
  applying this value, so unrelated shell allocator settings cannot change the
  experiment.

Both effective settings enter the service fingerprint. To inspect the combined
candidate without starting a process:

```bash
OMNI_VAE_OFFLOAD=1 OMNI_EXPANDABLE_SEGMENTS=1 \
  python3 omni-deployment/scripts/omni_control.py render sp2
```

Use the same flags for validation and startup after explicitly stopping the
previous service. Record this as a separate benchmark variant; a run with both
enabled cannot identify either setting's individual effect. Explicitly set both
to `0` for subsequent baseline runs if they were exported in the shell. This
package does not enable either option on the target host by itself.

## Reference image size policy

The versioned patch in `patches/reference-image-cap-v1.patch` adds explicit
image geometry to the pinned upstream source. Apply it only with a stopped
service, after distributing the package through Git:

```bash
"$OMNI_PYTHON" omni-deployment/scripts/omni_reference_patch.py apply \
  --source "$OMNI_FASTVIDEO_SRC" --project-root "$H3_OMNI_ROOT"
```

The helper checks the exact upstream commit, patch hash and all four source
hashes; modified or partially patched trees fail closed. The controller and
service verify this identity at startup and include it in the fingerprint.

`OMNI_REFERENCE_IMAGE_RESIZE_POLICY` accepts `native`, `cap` or `original`.
The controller's unset default remains native for compatibility. The target
env file selects cap480 for business deployment. With `cap`, set
`OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE` to an integer at least 32 (480 or 720 for
this evaluation). With native/original, leave it unset or set it to 0.

Cap modes never upscale small sources. The short edge aligns down to a multiple
of 32, then the long edge scales proportionally and aligns down; cap720 therefore
allows at most 704. Original uses the source dimensions subject only to that
alignment, without the upstream short-edge 2048 resize. Native retains the exact
upstream behavior. Cap and original explicitly disable the Qwen image processor's
second resize; video processing is unchanged. Numeric geometry logs allow the
report to compare requested policy and observed preprocessing. This does not
change output resolution or provide independent hard first-frame conditioning.

## Separate optimization experiments

`scripts/run_optimization_eval.py` fully validates the sealed nine-job business
protocol, then selects its three cap480 jobs. It retains the original prompt,
ordered references, plan hashes and seeds, requiring exactly one first request
and three warm requests with distinct seeds per case. Each case gets a fresh
owned SP2 service. Generation failure is recorded without retry; continuation
requires a confirmed clean stop. Lifecycle or ownership failure aborts the run.
The original protocol and `run_business_eval.py` remain unchanged.

Run one variant per new record and output directory. Supported variants are
`baseline` (`auto`, VAE offload), `cutlass`, `cudnn`, `vae-resident` (`auto`, VAE
offload disabled), and `decoder8` (`auto`, VAE offload, explicit `OMNI_DECODER8`
asset). All keep expandable segments enabled, TP1/SP2, cap480 and 8 PDD
forwards. Baseline/cutlass/cudnn/vae-resident use the explicit `OMNI_DECODER`
asset. Independently verify decoder provenance and layer count before running;
the runner records the full selected asset SHA, while a variant name does not
prove an 8-layer checkpoint. The controller also fingerprints the actual asset.

```bash
source omni-deployment/env.popos-5090.sh
"$OMNI_PYTHON" omni-deployment/scripts/run_optimization_eval.py \
  --protocol "$H3_OMNI_ROOT/run/omni/zero-business-001/protocol.json" \
  --variant cutlass --case case-01 \
  --record "$H3_OMNI_ROOT/run/omni/optimization-cutlass-smoke-001.json" \
  --output-root "$H3_OMNI_ROOT/outputs/omni-optimization-cutlass-smoke-001"
```

Omit `--case` for all three cases (12 requests). `--sp-profile 1` labels the
whole attempt diagnostic and ineligible for formal timing. `--stage-logging`
defaults to `1`; a `0` experiment requires its own matched baseline. Record paths
must be inside `run/omni`, output roots inside `outputs`, and neither may exist.
There is no automatic resume or overwrite. Results retain separate first/warm
phases. The record adds `optimization` metadata; the original business collector's
fixed baseline recipe is not a validator for these changed variants. Before
reporting speedups, compare the actual launch identity with this metadata and
review output quality independently.

## Lifecycle

The controller owns one service at a time, guarded by `run/omni/control.lock`.
`start` returns `STARTING`; `status` verifies both `/v1/models` identity and
`/health`. READY indicates engine readiness, not warmup or output quality.
The same running configuration is reused; switching SP1/SP2 or assets requires
an explicit stop. A different listener or process identity is never adopted.
The startup port probe uses `SO_REUSEADDR`, matching Linux server restart
behavior: prior connections in TIME_WAIT do not block a new service, while a
live listener still causes startup refusal. It does not enable `SO_REUSEPORT`.

```bash
python3 omni-deployment/scripts/omni_control.py stop
python3 omni-deployment/scripts/omni_control.py start sp2
python3 omni-deployment/scripts/omni_control.py status
```

State includes Linux PID start ticks, UID, process group, owner token, model name
and configuration fingerprint. Before reuse/signals all ownership fields are
checked. Logs get unique timestamp/token names and are not overwritten. Normal
stop sends TERM to the owning server so Uvicorn and the private Ray runtime can
clean up; the default stop timeout is 90 seconds. An in-flight request may take
longer. `stop --force` is an explicit emergency KILL of that owned process group;
after emergency shutdown inspect GPU processes for leftover Ray workers before
restarting. The package never runs a global `ray stop` or kills arbitrary GPU PIDs.

Automated operators can pass `stop --expected-owner-token TOKEN`, using the token
returned by their own `start`. Under the lifecycle lock, the controller rejects
missing state or a different token before checking process ownership or sending
any signal. This prevents stopping a replacement service between a prior status
check and the stop command. Manual `stop` without this option is unchanged.

SP2 creates Ray with `address="local"`, a private namespace/temp directory,
2 GPUs, 8 CPUs and a 4 GiB object store. It does not connect to an existing Ray
cluster. Before Ray initialization the launcher resolves the local node address and
sets it for both the driver and child workers; after initialization it verifies
the exact current-node resource and address. Ray can normalize loopback, so
changing only the driver after startup is insufficient. The HTTP service remains bound to loopback.
The object store cap is not a cap on all host memory: model loading,
pinned TE buffers, Python workers and activations must still be monitored.
Upstream `OpenAIServingEngine` already serializes requests with an asyncio lock.

## Ray health compatibility

The pinned upstream `/health` implementation assumes `worker.proc.is_alive()`;
Ray ActorHandles have no `proc`. `scripts/omni_ray_health.py` installs a scoped
property adapter in this server process without changing upstream files. MP
uses the original implementation. Ray checks the exact expected actor IDs with
the direct GCS actor-state lookup and requires every actor to be `ALIVE`.
Missing ranks, `DEAD`, `RESTARTING`, unknown state and query exceptions are
unhealthy. It never substitutes `/v1/models` for health.

The lookup is `ray._private.state.actors(hex_id)`, verified against Ray 2.49.1
source. This private API must be checked against the installed Ray version during
remote integration. Direct GCS lookup works without the dashboard and does not
queue behind a long GPU task. Like process `is_alive()`, it is a process-liveness
check; a completed smoke request remains necessary to test GPU execution.

## Verification performed locally

- Both final templates parsed through the pinned upstream `ServeConfig` parser.
- Python compilation passed.
- Twenty-one CPU-only service/controller tests passed: Ray handles without `proc`, failed/restarting or
  missing actors, GCS failure, shutdown/error state, MP fallback, PID reuse and
  owner/group/token mismatch refusal, venv executable symlink preservation,
  CUDA/media defaults and overrides, relative-path refusal, and project-tool
  precedence over unrelated executables in the inherited PATH, current Ray node
  resource matching, strict memory flags, allocator sanitization at validation
  and process startup, distinct fingerprints for all four memory variants, and
  expected-owner-token checks before process ownership checks and signals.
- Target-host native SP2 READY, five complete video/audio requests and normal
  shutdown passed; three-image 362 failed during the first forward. See validation
  evidence for exact limits. The low-memory variant passed three-image, image+video and image+audio
  at 362 frames plus image124. Formal performance is still under test. `validate` invokes actual config-to-engine conversion before startup.

Source basis: `examples/serving/openai_fasth3_omniref_rtx5090.yaml`,
`examples/inference/basic/basic_fasth3_omniref_pdd.py`, `fastvideo/api/schema.py`,
`fastvideo/entrypoints/cli/inference_config.py`,
`fastvideo/entrypoints/openai/serving_engine.py`, and
`fastvideo/worker/ray_utils.py` at the FastVideo pin above.
