# Reference image geometry patch v1

`reference-image-cap-v1.patch` targets FastVideo commit
`cdda1b7535fe32699335e39db82c1d4db21ceb88`. The accompanying JSON records the
patch SHA256 and the exact before/after SHA256 of its four source files. The
deployment helper checks this identity and applies the patch only on an
explicit `apply`; service startup never edits upstream source.

The fields live under `generator.pipeline.experimental` and are declared on
`MiniMaxH3PipelineConfig`:

| Policy | `h3_reference_image_resize_policy` | `h3_reference_image_max_short_edge` | Image processor resize |
| --- | --- | --- | --- |
| Released compatibility behavior | `native` (default) | `null` or omitted | Native default |
| Short-edge cap | `cap` | Integer >=32; 480 and 720 are accepted | Disabled |
| Source dimensions with alignment | `original` | `null` or omitted | Disabled |

The two non-native policies never enlarge either source dimension. They first
align the desired short edge downward to a multiple of 32, scale the long edge
by the same ratio, then align that edge downward. The remaining long-edge
rounding is less than 32 pixels, so preservation of aspect ratio is approximate.
This order preserves the upstream accepted aspect range of 1:4 through 4:1,
including small images. A source short edge below 32 is rejected. The 32-pixel
multiple comes from video VAE spatial compression (16) times DiT spatial patch
size (2). Input geometry is measured after EXIF orientation normalization.

| Source width × height | cap480 | cap720 | original |
| --- | --- | --- | --- |
| 1920 × 1080 | 832 × 480 | 1248 × 704 | 1856 × 1056 |
| 864 × 480 | 864 × 480 | 864 × 480 | 864 × 480 |
| 188 × 47 | 128 × 32 | 128 × 32 | 128 × 32 |

`original` means the source canvas with the required alignment reduction, not
byte-identical geometry or the released 2048-short-edge resizing. The default
`native` path retains the released geometry calculations exactly, including
their upscaling. The three requested comparison variants are cap480, cap720,
and original; native is a separate compatibility baseline.

Prepared images are passed to both the reference VAE and image conditioning.
For cap/original, conditioning explicitly sets `do_resize=False` so Qwen's
minimum pixel setting cannot enlarge small prepared images again. The
conditioning memo key includes the policy. Video-reference processing and
target output dimensions, frame count, and FPS are unchanged. The
[Transformers 5.15.1 Qwen2-VL processor source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
supports `do_resize`; target-runtime synthetic processor verification remains
an independent deployment check.

After cleanly stopping the owned service, use the deployment helper:

```sh
"$OMNI_PYTHON" omni-deployment/scripts/omni_reference_patch.py check \
  --source "$OMNI_FASTVIDEO_SRC" --project-root "$H3_OMNI_ROOT"
"$OMNI_PYTHON" omni-deployment/scripts/omni_reference_patch.py apply \
  --source "$OMNI_FASTVIDEO_SRC" --project-root "$H3_OMNI_ROOT"
```

The portable source/patch regression test uses temporary copies and geometry
stubs. It never imports model dependencies or opens media:

```sh
OMNI_FASTVIDEO_SRC="$OMNI_FASTVIDEO_SRC" "$OMNI_PYTHON" -B -m unittest discover \
  -s omni-deployment/tests -p test_omni_reference_image_cap.py -v
```

Each non-native image emits `H3_REFERENCE_IMAGE_GEOMETRY` with the policy,
source width/height, prepared width/height, requested cap (or `None`), and
alignment multiple. These fields contain no path, prompt, or pixels. For
runtime proof of the image encoder contract, compare its numeric
`image_grid_thw` against the prepared dimensions divided by the configured
patch size (16); no media rendering or pixel logging is needed.

Validation: 11 geometry/configuration/propagation/processor-contract tests and
six patch-identity tests pass. The four temporary source copies also pass the
upstream pinned `pre-commit run --files` hooks, including yapf, ruff, codespell,
and mypy, using Python 3.12. The source fixture contains only the four changed
files and upstream tool configuration; this is a local static and contract
check, not a model execution or a full FastVideo integration test.
