# Business request evaluation

The former synthetic/Tears of Steel case plans were removed on 2026-10-21 at the user's request. They are no longer part of acceptance or current latency reports.

The current evaluation uses the same authorized real business requests across three reference-image preprocessing policies: short-edge cap 480, cap 720, and original source size. Output remains the original business 480p portrait contract (480×832, 362 frames, 24 fps). Only reference image geometry changes. Caps never upscale smaller sources; geometry aligns down to the model's 32-pixel grid (720 becomes at most 704). The original policy also aligns down to that grid and disables the upstream 2048-pixel upscale.

Use SP2 on GPUs 0,1, released eight-forward PDD/CFG1, the 36-layer decoder, VAE offload and expandable allocation, with diagnostic SP profiling disabled. Each case/policy gets a fresh service, one explicit first request (seed 9000), and three warm measurements (seeds 1234, 5678, 9012). Keep the original prompt and ordered references identical across policies. Rotate policy order across cases. Report fresh-service first-request and warm timing separately; three samples per cell do not establish a tail-latency SLA.

Private request payloads, prompts, media, identifiers and plans belong in ignored runtime directories, never in this public repository. Record source provenance and media SHA256 locally. Report any source API capability that cannot be represented by the local Ref2VA pipeline (including independent hard first-frame conditioning) explicitly; mapping that image as a reference is not an equivalent guarantee.

Runtime plans, service fingerprints, source patch hashes, processed geometry observations, validation manifests and resource samples must agree before a timing enters the business report. Failed, interrupted and unrun requests remain distinct. Videos and human-only quality frames go to a new output directory; agents do not open media.

After placing the sealed private inputs and protocol under the deployment root,
invoke the committed runner with new record and output paths:

```bash
cd /home/isaac/workdir/h3
source omni-deployment/env.popos-5090.sh
"$OMNI_PYTHON" omni-deployment/scripts/run_business_eval.py \
  --protocol "$H3_OMNI_ROOT/run/omni/<run-id>/protocol.json" \
  --record "$H3_OMNI_ROOT/run/omni/<run-id>.json" \
  --output-root "$H3_OMNI_ROOT/outputs/omni-<run-id>" \
  --project-root "$H3_OMNI_ROOT"
```

The runner requires the service to be stopped before each job. It verifies plan,
prompt and reference identities, owns only the processes it starts, and stops
subsequent jobs if graceful cleanup cannot be confirmed. It never resumes or
overwrites an existing run. A failed generation stays recorded as a failure;
there is no automatic regeneration to improve a sample count.
