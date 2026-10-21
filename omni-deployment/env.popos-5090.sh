# Source from a shell on popos-5090 after verifying the pinned assets.
# This file selects the dedicated Omni installation; it starts no processes.
export H3_OMNI_ROOT=/home/isaac/workdir/h3
export OMNI_PYTHON="$H3_OMNI_ROOT/envs/omni/bin/python"
export OMNI_FASTVIDEO_SRC="$H3_OMNI_ROOT/src/FastVideo"
export OMNI_MODEL_DIR="$H3_OMNI_ROOT/models/FastH3-Omni-NVFP4-QAD-f01e6a23"
export OMNI_DECODER="$OMNI_MODEL_DIR/vae_decoder_nvfp4/decoder-nvfp4.safetensors"
export OMNI_ADALN_TABLE="$OMNI_MODEL_DIR/tables/omni-qad-8nfe.pt"
export OMNI_ASSET_MANIFEST="$H3_OMNI_ROOT/run/omni/verified-assets.json"
export OMNI_INPUT_DIR="$H3_OMNI_ROOT/inputs"
export OMNI_OUTPUT_DIR="$H3_OMNI_ROOT/outputs/omni"
# Verified target low-memory recipe; set either flag to 0 before sourcing for native experiments.
export OMNI_VAE_OFFLOAD="${OMNI_VAE_OFFLOAD:-1}"
export OMNI_EXPANDABLE_SEGMENTS="${OMNI_EXPANDABLE_SEGMENTS:-1}"
# Business reference geometry; requires the explicit versioned source patch.
export OMNI_REFERENCE_IMAGE_RESIZE_POLICY="${OMNI_REFERENCE_IMAGE_RESIZE_POLICY:-cap}"
if [ "$OMNI_REFERENCE_IMAGE_RESIZE_POLICY" = cap ]; then
  export OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE="${OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE:-480}"
else
  export OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE="${OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE:-0}"
fi
export CUDA_HOME="$H3_OMNI_ROOT/envs/omni/lib/python3.12/site-packages/nvidia/cu13"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$H3_OMNI_ROOT/envs/omni/bin:$CUDA_HOME/bin:$H3_OMNI_ROOT/.tools/media/bin:$PATH"
