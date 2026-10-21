# Omni AdaLN preparation and table-only verification

`prepare_omni_adaln.py` reuses the public Omni driver at FastVideo commit
`cdda1b7535fe32699335e39db82c1d4db21ceb88`. It never downloads models, installs
dependencies or starts a remote connection. Run it in the dedicated FastVideo
environment after the complete local QAD snapshot and native kernels are ready.

## Request file

Paths are absolute or relative to the JSON file. Reference order is preserved.
Use the project's approved benchmark assets. Every request writes a separate MP4.
The `phase` field is a reporting label; there is no hidden warmup or repeat.

```json
{
  "requests": [
    {
      "id": "image-no-reference-audio",
      "phase": "prepare",
      "prompt": "The person in <Picture 1> turns and looks at the camera.",
      "references": [{"media_type": "image", "source": "reference.png"}],
      "num_frames": 124,
      "seed": 1234
    },
    {
      "id": "image-with-reference-audio",
      "phase": "prepare",
      "prompt": "The person in <Picture 1> speaks, using the voice in <Audio 1>.",
      "references": [
        {"media_type": "image", "source": "reference.png"},
        {"media_type": "audio", "source": "voice.wav"}
      ],
      "num_frames": 124,
      "seed": 1235
    }
  ]
}
```

Preparation requires these two modality classes. A video with a soundtrack is
also supported, but does not replace the explicit audio coverage check. Add
actual video-reference requests for functional coverage. Accepted `num_frames`
values are 124 and 362. Sampling remains 832×480, 24 fps, CFG 1 and the released
eight-forward PDD contract. This deliberately does not use the old H3 serving
`NFE+1` conversion.

## Prepare once on SP1

```bash
python scripts/prepare_omni_adaln.py \
  --mode prepare \
  --fastvideo-root /home/isaac/workdir/h3/src/FastVideo \
  --model-dir /absolute/path/to/FastH3-Omni-NVFP4-QAD \
  --decoder /absolute/path/to/FastH3-Omni-NVFP4-QAD/vae_decoder_nvfp4/decoder-nvfp4.safetensors \
  --requests-json /absolute/path/to/prepare-requests.json \
  --table /absolute/path/to/tables/omni-qad-8nfe.pt \
  --output-dir /absolute/path/to/outputs/adaln-prepare-001
```

The output directory and published table must be new. The wrapper explicitly
passes the same QAD directory as both model and base, avoiding the driver's
default BF16 MiniMax base-component download path. It uses `CACHE=1` plus `DUMP`
and clears inherited TABLE settings before importing FastVideo. The pinned
loader stages on CPU, keeps the AdaLN projections in pinned host memory and
copies one projection to GPU on a miss. This still needs free host RAM; no
memory-capacity guarantee is implied.

One generator handles all requests, accumulating their clock keys. Every
request must generate valid video and audio. The final table must have all 50
blocks, the same complete key set in each block, at least eight keys per block,
and finite modulation tensors with the checkpoint's full projection width.
Embedding-key dtype and modulation dtype are recorded separately: this release
uses FP32 timestep embeddings with BF16 projection results, so equality would
incorrectly reject a valid table.
Publication uses an atomic no-replace hard link after all checks pass.

An interrupted or failed run leaves `adaln_table.partial.pt` and its diagnostic
`.inputs` in the run directory. It does **not** publish the requested final
table. Do not treat a partial file as a ready artifact. A new prepare run needs
a new output directory. Automatic resume/merge is intentionally unsupported.

The key includes unique timestep values, embedding shape and dtype. It does not
include frame count. The fixed ladder permits 124→362 reuse for the same clock
sets, while adding reference audio can add a different clock key. Preparation
therefore covers both modality classes, followed by fresh table-only checks.

## Verify a new process, then SP2

```bash
python scripts/prepare_omni_adaln.py \
  --mode validate --num-gpus 2 \
  --fastvideo-root /home/isaac/workdir/h3/src/FastVideo \
  --model-dir /absolute/path/to/FastH3-Omni-NVFP4-QAD \
  --decoder /absolute/path/to/FastH3-Omni-NVFP4-QAD/vae_decoder_nvfp4/decoder-nvfp4.safetensors \
  --requests-json /absolute/path/to/validation-requests.json \
  --table /absolute/path/to/tables/omni-qad-8nfe.pt \
  --output-dir /absolute/path/to/outputs/adaln-validate-sp2-001
```

First validate on SP1 (`--num-gpus 1`), then SP2. Include 124-frame and 362-frame
image-only, image+audio, multiple-image and video-reference requests. The same
process can run explicit warmup and timed request rows, with distinct seeds and
IDs. SP2 uses the pinned driver's Ray, TP1/SP2, replicated NVFP4 and parallel-VAE
settings. GPU and host-memory monitoring remains external to this wrapper.

## Outputs and measurement limits

- `run.json`: pinned source revision, requested model revision, actual local
  paths, contract/request hashes, selected environment, lifecycle status and
  final table hash. The model revision must additionally be verified against
  the downloader's manifest; a local folder name is not proof of a revision.
- `requests.jsonl`: append-only passed/failed request records, seed, phase,
  full `generate()` wall time including writing the local MP4, official
  generation time, official rank-0 peak allocation, and ffprobe results.
- One MP4 per request. Checks require the requested dimensions/frame count,
  24 fps, 32-kHz stereo audio, and audio/video durations within 0.20 seconds.
  ffprobe decoding/validation time is recorded separately from generation.
- `configured_transformer_forwards=8` comes from the validated PDD contract.
  `measured_transformer_forwards` and per-rank peak allocation are `null`:
  this wrapper does not invent measurements absent from the public result.

Do not combine warmup, preparation or failed requests into a successful warm
latency average. New prompt plus fixed references only reuses the independent
image-latent cache: the pinned conditioning cache includes the prompt and
disables its memo for video references. Report cache conditions explicitly.

This script performs no visual quality judgement; inspect the generated clips
and sound before accepting a performance result.
