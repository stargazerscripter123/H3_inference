# FastH3 Omni on popos-5090

Reviewable deployment package for the rebuilt dual-RTX-5090 host. It prepares a
matching AdaLN table, owns one persistent local service, and records validated
image/video/audio generation requests and sequential benchmarks. SP2 means
sequence parallelism with replicated NVFP4 weights and parallel VAE; TP stays 1.
Two 32-GB cards do not become one 64-GB memory pool.

**Status on 2026-10-21:** The pinned environment, model assets, AdaLN table and
SP2 generation path have been verified. At the user's request, the former test
case plans and video/statistics outputs have been removed from current
acceptance. Evaluation now uses real business requests across reference-image
cap480, cap720 and original-size policies, with fixed 480p output. These new
business results are pending; historical synthetic timings do not establish
business latency or quality. See [the evaluation contract](benchmarks/README.md).
Environment and immutable model evidence remain in [ENVIRONMENT.md](ENVIRONMENT.md).

## Fixed source and model versions

| Item | Version or requirement | Verification status |
| --- | --- | --- |
| FastVideo | `cdda1b7535fe32699335e39db82c1d4db21ceb88` | Local source inspected; launcher checks target checkout HEAD |
| Native FastH3 Omni NVFP4 QAD | `f01e6a23ac6066aa9eb622cf94b6259442e70ec2` | All 47 files verified against pinned source identities; see environment/verified-model-assets.json |
| Sampling contract | Ref2VA, released 8-forward PDD, CFG 1, 24 fps | Client/schema CPU checks passed; no legacy `NFE+1` conversion |
| Python | Project-managed 3.12.15, including development headers | Installed in the dedicated target venv |
| Torch / CUDA wheels | Pinned source requires Torch 2.12.0; target installation uses cu130 | Installed; targeted native GPU tests passed; resolved freeze recorded |
| FastVideo kernel | Build from the fixed FastVideo source; PyPI 0.3.5 predates the required sparse FP4 API | Built from pinned source for sm120a; targeted numerical tests passed |
| CUDA build tools | NVCC 13.0.88 and CCCL 13.0.85 | Installed with env-local development-library links; source build passed |
| Ray | Source requires `>=2.49.1`; health adapter reviewed against 2.49.1 | Installed Ray2.59.0; private GCS ALIVE probe passed; native SP2 READY, five media requests and normal shutdown passed; memory variant under test |
| Optional FA4 stack | Source pins FlashAttention `14c377950125c70b7a9dabf9c561fca53715ac7d`, CuTe DSL `4.6.0.dev0`, quack `0.5.3` | These are upstream requirements, not an installed lock; serving recipe sets `FASTVIDEO_FA4=0` |

The new host reports Pop!_OS 24.04, driver 595.104.02, about 125 GiB RAM, GPU0
PCIe x16 and GPU1 x8. GPU topology remains NODE with peer access unavailable.
SP2 speedup must therefore be measured on this machine. Historical H3/vLLM TP2
results are not measurements of this Omni executor, model or operating system.

中文操作入口：[RUNBOOK.zh-CN.md](RUNBOOK.zh-CN.md)。

## Package contents

| File | Role |
| --- | --- |
| `configs/fasth3-omni-sp1.template.json` | Complete single-GPU MP serve configuration; GPU1 by default |
| `configs/fasth3-omni-sp2.template.json` | Complete two-GPU Ray serve configuration; GPUs 0,1 by default |
| `scripts/omni_control.py` | Render, validate, start, identify, reuse and stop this package's service |
| `scripts/omni_service.py` | Pinned-source bootstrap and private, bounded Ray startup |
| `scripts/omni_ray_health.py` | Scoped correction for upstream Ray ActorHandle liveness checking |
| `scripts/prepare_omni_adaln.py` | Prepare and atomically publish a complete table; validate in a fresh process |
| `scripts/run_omni.py` | One HTTP video job, reference hashes, download, ffprobe validation and manifest |
| `scripts/bench_omni.py` | Explicit sequential warmup/measured plan with separate scenario/cache summaries |
| `scripts/monitor_omni_resources.py` | Read-only Linux GPU/host memory and scoped process RSS sampling; JSONL and peak summary |
| `tests/test_omni_service_control.py` | CPU tests for ownership, venv path handling and Ray liveness adapter |
| `tests/test_omni_adaln.py` | Regression tests for FP32 embedding keys, BF16 modulation and table integrity |
| `tests/test_omni_resource_monitor.py` | CPU parser, process scope/PID reuse, and signal-summary checks |
| `scripts/test_run_omni.py` | CPU tests for request contracts, mocked job lifecycle, validation and reporting |
| `SERVICE.md` | Service configuration, lifecycle, environment and compatibility details |
| `scripts/prepare_omni_adaln.md` | AdaLN request format, commands and publication/validation rules |

## Deployment order

Run the following stages on `popos-5090` after the package has been reviewed and
distributed. Commands below assume this directory is installed as
`/home/isaac/workdir/h3/omni-deployment`. Preparation, SP1 and SP2 runs must be
sequential: stop any service before starting direct AdaLN preparation/validation.

### 1. Complete the dedicated environment and immutable assets

The deployment root is `/home/isaac/workdir/h3`, source is `src/FastVideo`,
environment is `envs/omni`, and uv is `.bootstrap/bin/uv`. Use the pinned source's
installation requirements, including the x86_64/SM120 FP4 kernels; record the
successful installation commands and resolved package versions. Do not treat
package import success as proof that a CUDA kernel can run.

Use the project-managed Python, since the rebuilt host's system Python lacks
development headers. The final environment is `envs/omni` and retains its venv
interpreter path even though that executable links to the managed Python.

For both `validate` and `start`, the controller automatically sets `CUDA_HOME`
to `<venv>/lib/python3.12/site-packages/nvidia/cu13` and `CUDACXX` to its
`bin/nvcc` (the target has NVCC 13.0.88). Its child `PATH` begins with the venv's
`bin`, CUDA's `bin`, then `<deployment-root>/.tools/media/bin`, followed by the
caller's `PATH`. The media directory contains the project's `ffmpeg` and `ffprobe`.
Starting the service from a new shell therefore needs no manual CUDA or media
exports. `OMNI_CUDA_HOME` and `OMNI_MEDIA_BIN` accept absolute path overrides.
The default layout is for the deployment's Python 3.12 environment; use
`OMNI_CUDA_HOME` when placing CUDA elsewhere. These settings are part of the
service fingerprint, so changing them requires an explicit stop/start.

Verify source HEAD, Torch/CUDA versions, both GPU identities, available host/GPU
memory, `ffmpeg`/`ffprobe`, and imports of the actual NVFP4/VSA extensions. Capture
the final dependency freeze, kernel build provenance and fresh GPU topology in
the deployment record. The current bootstrap/download process is separate from
this package; this package does not install dependencies or download weights.

The model path must be a complete local QAD snapshot or a verified composed
directory. Required components are `transformer_ref/nvfp4_weights.safetensors`,
`scheduler`, `audio_scheduler`, `text_encoder`, `processor`, `tokenizer`, `vae`,
`audio_vae`, and `fastvideo_inference.json`. Keep the released 36-layer packed
decoder for the baseline. Record repository revisions, component provenance and
file hashes; a directory name is not revision verification. Do not silently
substitute a different base model or download the full BF16 base.

Set actual verified paths, replacing every `/absolute/path/...` placeholder:

For the known `popos-5090` layout, `source /home/isaac/workdir/h3/omni-deployment/env.popos-5090.sh`
selects the dedicated interpreter, source/model/table paths and local tools.
It starts no processes and does not verify assets; complete the checks below
before using the selected paths. Other layouts can set the variables explicitly.

```bash
export H3_OMNI_ROOT=/home/isaac/workdir/h3
export OMNI_FASTVIDEO_SRC="$H3_OMNI_ROOT/src/FastVideo"
export OMNI_PYTHON="$H3_OMNI_ROOT/envs/omni/bin/python"
export OMNI_MODEL_DIR=/absolute/path/to/complete-or-composed-QAD-model
export OMNI_DECODER=/absolute/path/to/vae_decoder_nvfp4/decoder-nvfp4.safetensors
export OMNI_ADALN_TABLE=/absolute/path/to/tables/omni-qad-8nfe.pt
export OMNI_INPUT_DIR="$H3_OMNI_ROOT/inputs"
export OMNI_OUTPUT_DIR="$H3_OMNI_ROOT/outputs/omni"
cd "$H3_OMNI_ROOT/omni-deployment"
git -C "$OMNI_FASTVIDEO_SRC" rev-parse HEAD
"$OMNI_PYTHON" --version
"$H3_OMNI_ROOT/.tools/media/bin/ffprobe" -version
```

After a verified asset manifest exists, set `OMNI_ASSET_MANIFEST` to its absolute
path. The controller fingerprints it, the contract, table and decoder. It checks
component presence but does not independently hash every model shard.

### 2. Prepare the matching AdaLN table on SP1

The direct preparation and client programs run outside the controller. For
those commands only, give the current shell the same tools:

```bash
export CUDA_HOME="${OMNI_CUDA_HOME:-$H3_OMNI_ROOT/envs/omni/lib/python3.12/site-packages/nvidia/cu13}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$H3_OMNI_ROOT/envs/omni/bin:$CUDA_HOME/bin:${OMNI_MEDIA_BIN:-$H3_OMNI_ROOT/.tools/media/bin}:$PATH"
```

Follow `scripts/prepare_omni_adaln.md` to write a preparation request file using
approved reference media. Cover both a visual reference without explicit audio
and a visual reference with explicit audio. Use 124 frames initially. Reference
paths must exist on the host. Preparation requires both classes because their
clock keys can differ; a video soundtrack does not satisfy the explicit-audio
coverage check.

```bash
CUDA_VISIBLE_DEVICES=1 "$OMNI_PYTHON" scripts/prepare_omni_adaln.py \
  --mode prepare \
  --fastvideo-root "$OMNI_FASTVIDEO_SRC" \
  --model-dir "$OMNI_MODEL_DIR" \
  --decoder "$OMNI_DECODER" \
  --requests-json /absolute/path/to/prepare-requests.json \
  --table "$OMNI_ADALN_TABLE" \
  --output-dir "$H3_OMNI_ROOT/outputs/adaln-prepare-001"
```

The output directory and final table path must not already exist. One generator
accumulates keys, and publication occurs only after all output and table checks
pass. Monitor host memory throughout CPU staging and pinned AdaLN projection
storage. An interrupted run leaves diagnostic partial files; it does not publish
a ready table or automatically resume.

Then run a fresh `--mode validate --num-gpus 1` process with a new output
directory and a validation request file. Cover 124 and 362 frames, image-only,
image+audio, multiple-image and video-reference cases before proceeding. A
separate direct `--num-gpus 2` validation is available, but uses the upstream
driver's Ray setup. The persistent service below adds explicit Ray resource
bounds and should be used for the final SP2 acceptance and timing runs.

### 3. Validate and smoke-test the persistent SP1 service

```bash
"$OMNI_PYTHON" scripts/omni_control.py render sp1
"$OMNI_PYTHON" scripts/omni_control.py validate sp1
"$OMNI_PYTHON" scripts/omni_control.py start sp1
"$OMNI_PYTHON" scripts/omni_control.py status
```

`start` returns `STARTING`; wait for `status` to report `READY` and inspect the
service log before submitting work. Readiness verifies model identity and
engine health; it does not warm kernels or validate media. The default endpoint
is `http://127.0.0.1:8108`. Only local references under `OMNI_INPUT_DIR` are
allowed. State and rendered configuration are in `run/omni`; unique service
logs are in `logs/omni` under the deployment root.

With a real approved image at the example path:

```bash
"$OMNI_PYTHON" scripts/run_omni.py \
  --prompt 'The person in <Picture 1> turns and looks at the camera.' \
  --image "$OMNI_INPUT_DIR/omni/reference.png" \
  --frames 124 --seed 1234 \
  --output-root "$H3_OMNI_ROOT/outputs/omni-smoke-sp1"
```

Repeat the functional modality matrix at 124 frames, then 362 frames. Visually
inspect clips and listen to their audio. Automatic validation requires exact
resolution/frame count/24 fps, one video stream, one 32-kHz stereo audio stream,
and matching durations; it cannot judge identity, motion, lip sync or quality.

HTTP reference order must be all images, then videos, then audio, with order
preserved within each type. The client rejects interleaving because the pinned
HTTP adapter would reorder it. Limits are 9 images, 3 videos, 3 audio references,
12 total, and at least one image or video. It accepts the frame grid `17*n+5`
from 124 through 362, uses eight PDD forwards, and performs no preprocessing or
upload. `--server-paths` is only for already resident absolute server paths;
it disables client-side reference existence/hash checks.

### 4. Stop SP1, then test SP2 on both cards

```bash
"$OMNI_PYTHON" scripts/omni_control.py stop
"$OMNI_PYTHON" scripts/omni_control.py validate sp2
"$OMNI_PYTHON" scripts/omni_control.py start sp2
"$OMNI_PYTHON" scripts/omni_control.py status
```

Repeat the same smoke and modality cases with the same assets, table, decoder
and sampling settings. Verify both ranks execute, `/health` works with the
installed Ray version, and shutdown releases the owned GPU workers. Capture
per-GPU memory, host memory and transfer behavior externally. Upstream reported
peak allocation is not a substitute for independently measured peaks on both
cards.

SP2 creates a private local Ray runtime with 2 GPUs, 8 CPUs and a 4-GiB object
store. The object-store limit does not bound total host memory. The health
adapter checks every expected actor's GCS state and fails closed; it does not
query the model list as a substitute for health. See `SERVICE.md` for its private
Ray API compatibility requirement and process ownership safeguards.

### 5. Measure explicit workload and cache conditions

Build one JSON plan per repeatable benchmark set. This format is for the HTTP
benchmark and differs from the AdaLN preparation request format. Example using
an already resident image; duplicate measured rows as needed with explicit
new prompts/seeds and new media content for a `warm-new` test:

```json
{
  "requests": [
    {
      "name": "single-image-warmup",
      "case_id": "single-image",
      "phase": "warmup",
      "cache_condition": "unspecified",
      "prompt": "The person in <Picture 1> looks toward the camera.",
      "seed": 1234,
      "frames": 362,
      "references": [{"type": "image", "source": "/home/isaac/workdir/h3/inputs/omni/reference.png"}]
    },
    {
      "name": "single-image-new-prompt-1",
      "case_id": "single-image",
      "phase": "measured",
      "cache_condition": "warm-fixed-ref",
      "prompt": "The person in <Picture 1> smiles and turns to the left.",
      "seed": 1235,
      "frames": 362,
      "references": [{"type": "image", "source": "/home/isaac/workdir/h3/inputs/omni/reference.png"}]
    }
  ]
}
```

```bash
"$OMNI_PYTHON" scripts/bench_omni.py \
  --plan /absolute/path/to/benchmark-requests.json \
  --output-root "$H3_OMNI_ROOT/outputs/omni-bench-sp2" \
  --threshold 15
```

Use separate `case_id` values for single image, multiple images, image+audio,
video and mixed-reference scenarios. Summaries group by case, cache label and
geometry. Omitted case IDs are grouped as `all-unspecified`. Run the identical
plan on SP1 and SP2, with separate output roots and equivalent warmup. The
default SP1 baseline uses GPU1; a separate GPU0 run can distinguish the effect
of the unequal PCIe links. Switching requires explicit stop/start.

Cache labels describe the operator's experiment; the API exposes no cache-hit
proof. `cold` requires an actual fresh process/appropriate cold setup, which the
benchmark does not create. `warm-new` requires new content and prompts;
`warm-fixed-ref` uses new prompts with fixed references;
`warm-full-conditioning` reuses the complete conditioning. The pinned
conditioning memo includes the prompt and is disabled for video references.
Do not label a new prompt with fixed references as full-conditioning reuse.

`client_to_media_s` starts immediately before POST and ends after the complete
MP4 download. It includes polling delay, queueing and download, excludes input
hashing/staging before submission, and excludes local ffprobe/hash validation
after download. `client_to_validated_s` includes that final validation.
`server_reported_inference_s` is retained separately; the pinned async API does
not expose a complete server request timer. Do not compare these scopes as if
they were interchangeable. The default polling interval is 0.25 seconds.

The benchmark submits sequentially with explicit warmup rows and no hidden
retries/restarts/cache clearing. It stops on the first failed/unknown request
because that job may still be running. Failures remain in the valid-on-time
denominator; unrun requests are reported separately. Percentiles explicitly
cover successful requests only. A client timeout does not cancel server work.
Inspect the retained job ID and manifest before any manual retry.

### External resource monitoring during smoke tests and benchmarks

Run the standard-library monitor in another terminal on the target host before
the smoke/benchmark request. Keep separate output files for SP1/SP2 and each run:

```bash
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/monitor_omni_resources.py" \
  --project-root "$H3_OMNI_ROOT" --interval 1 --duration 1800 \
  --output "$H3_OMNI_ROOT/outputs/resources/sp1-screen-001.jsonl"
```

Omit `--duration` to sample until Ctrl-C or a SIGTERM sent specifically to this
monitor. Its signal handlers only stop its own loop; the monitor never signals
the observed processes or changes GPU/service settings. A running `nvidia-smi`
query has a five-second timeout, so finalization may wait for that query. Both
normal duration expiry and handled SIGINT/SIGTERM write a final JSONL summary
and sibling `sp1-screen-001.summary.json`. SIGKILL, power loss, or a full filesystem
cannot guarantee that final summary. Existing evidence files are not overwritten.

Each sample records UTC time and monotonic elapsed time, physical GPUs 0 and 1
with UUID/index, GPU/memory utilization percentages, device memory used/total
in MiB, Linux `MemAvailable`/`SwapFree` and totals in bytes, and selected process
RSS in bytes. **GPU memory is whole-device `nvidia-smi` usage, including display
and unrelated processes; it is not PyTorch allocated/reserved memory.** Summaries
report sampled per-device peaks, minimum host available/free memory, maximum
total-minus-available memory/swap usage, and per-process/aggregate selected RSS
peaks. Summed RSS can count shared pages more than once and is not total physical
memory consumption. Sampling can miss brief peaks; collection duration is recorded
to show overhead or an interval the machine could not maintain.

Process RSS is limited to the same user and executable/script paths under this
project's `envs/omni` or `omni-deployment`. The script inspects only executable
and script-path tokens, stopping before application arguments or Python `-c`
code. It never reads `/proc/*/environ`, credentials, or saves full command lines.
RayWorkerWrapper actors require an identified project ancestor, or an explicitly
verified `--ray-actor-pid PID` (repeatable). Explicit actor PIDs must already exist;
their `/proc` start time is bound on startup and checked against reuse. An unrelated
Ray actor with the same process title is excluded. Actors without a surviving
identifiable ancestor require the explicit PID option, so review process coverage
instead of assuming every worker was discovered. Run the monitor with the same
user as the service; it does not escalate access.

Query failures and missing RSS are recorded as unavailable/partial evidence,
not zero usage. Per-process exit races and discovery gaps remain in the JSONL;
the summary exposes skipped/raced counts and samples with no selected processes.
The exit code is 0 for samples without source errors, 2 when samples are absent
or source errors occurred, and 1 for setup/output failures. A partial RSS sum is
only the sum of successfully read selected processes. Keep these files beside
the client reports and service logs; the monitor does not link resource samples
to individual request IDs or judge generation quality.

### 6. Acceptance and normal shutdown

Accept the deployment only after the actual environment/assets are recorded,
the fresh table-only and persistent-service modality cases pass, both SP2 ranks
are observed, and repeated SP1/SP2 measurements have valid media and separately
reported cache conditions. Keep manifests, rendered config, service logs and
monitoring alongside each result. The 15-second threshold is a measurement
target, not a property guaranteed by the package.

```bash
"$OMNI_PYTHON" scripts/omni_control.py stop
```

Normal stop targets only the recorded owned service and lets its private Ray
runtime unwind. It never invokes global `ray stop`. If the 90-second normal
timeout expires, inspect active work; `stop --force` explicitly kills the owned
process group and requires checking for leftover Ray workers before restart.

## Local verification available to reviewers

From the package directory:

```bash
python3 -m unittest discover -s tests -p 'test_omni_service_control.py' -v
python3 -m unittest discover -s scripts -p 'test_run_omni.py' -v
python3 -m unittest discover -s tests -p 'test_omni_resource_monitor.py' -v
python3 -m py_compile scripts/omni_control.py scripts/omni_service.py scripts/omni_ray_health.py scripts/prepare_omni_adaln.py scripts/run_omni.py scripts/bench_omni.py
```

Observed locally: 19 lifecycle/health/toolchain, 2 Ray startup, 5 AdaLN,
6 resource monitor and 16 client/benchmark tests passed (48 total);
Python compilation passed; both final JSON templates parsed with the pinned
upstream `ServeConfig` parser. The client tests mock HTTP/media operations and
do not execute GPUs. Target `omni_control.py validate` additionally checks the
real environment's config-to-engine conversion before service startup. No
GPU-output, quality, memory, latency or long-running-service claim follows from
these local checks.
