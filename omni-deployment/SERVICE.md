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
The service binds only `127.0.0.1`; only files within `OMNI_INPUT_DIR` may be
submitted as local references (default `/home/isaac/workdir/h3/inputs`). The
source checkout HEAD must match the pin.
Inherited `FASTVIDEO_*`, `NCCL_*`, and `RAY_*` overrides are removed before
applying this recipe so another experiment cannot silently change its backend.

## Lifecycle

The controller owns one service at a time, guarded by `run/omni/control.lock`.
`start` returns `STARTING`; `status` verifies both `/v1/models` identity and
`/health`. READY indicates engine readiness, not warmup or output quality.
The same running configuration is reused; switching SP1/SP2 or assets requires
an explicit stop. A different listener or process identity is never adopted.

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

SP2 creates Ray with `address="local"`, a private namespace/temp directory,
2 GPUs, 8 CPUs and a 4 GiB object store. It does not connect to an existing Ray
cluster. The object store cap is not a cap on all host memory: model loading,
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
- Eight CPU-only tests passed: Ray handles without `proc`, failed/restarting or
  missing actors, GCS failure, shutdown/error state, MP fallback, PID reuse and
  owner/group/token mismatch refusal, and venv executable symlink preservation.
- GPU load, Ray startup/shutdown, API smoke and performance require target-host
  verification. `validate` additionally invokes pinned upstream config-to-engine
  conversion with the real target environment before any service startup.

Source basis: `examples/serving/openai_fasth3_omniref_rtx5090.yaml`,
`examples/inference/basic/basic_fasth3_omniref_pdd.py`, `fastvideo/api/schema.py`,
`fastvideo/entrypoints/cli/inference_config.py`,
`fastvideo/entrypoints/openai/serving_engine.py`, and
`fastvideo/worker/ray_utils.py` at the FastVideo pin above.
