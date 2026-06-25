# syntax=docker/dockerfile:1.6
#
# Confidential Kimi K2.6 with DCP (Decode Context Parallel) + Eagle 3.1 + fp8 KV cache.
#
# Based on vLLM v0.23.0 with patches from:
#   - PR #40750: [Attention] Enable TRITON_MLA MTP full CUDA graphs for Kimi on Blackwell
#   - PR #40611: [SpecDecode] Allow draft-specific attention backend and KV dtype
#
# Features:
#   - TRITON_MLA attention backend with DCP=8 and fp8 KV cache
#   - Eagle 3.1 speculative decoding (draft model runs without DCP)
#   - ~16x KV cache capacity vs non-DCP (39.9M tokens vs 2.4M tokens)
#   - RunAI model streamer for faster weight loading
#
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.23.0-ubuntu2404@sha256:662e4975c5c9947f8723f4d8f438145971361a480a2ade1919bb9462a9f24088
FROM ${VLLM_BASE_IMAGE}

COPY patches/ /tmp/tinfoil-patches/

# Step 1: Apply PR #40750 patch with --forward (applies what it can, skips failures)
RUN set -eux; \
    cd /; \
    patch -p1 --no-backup-if-mismatch --fuzz=0 --forward < /tmp/tinfoil-patches/0001-pr40750-dcp-fp8-triton-mla.patch || true; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '*.rej' -delete 2>/dev/null || true

# Step 2: Replace triton_mla.py entirely with PR version (complete rewrite for DCP support)
RUN set -eux; \
    cp /tmp/tinfoil-patches/triton_mla.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla.py; \
    cp /tmp/tinfoil-patches/triton_mla_tuning.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla_tuning.py

# Step 3: Apply manual patches for hunks that failed to apply against v0.23.0
# (assert k_scale removal, speculative.py draft fields, flashinfer.py maybe_override_cp)
RUN set -eux; \
    python3 /tmp/tinfoil-patches/apply_manual_patches.py

# Step 4: Apply DCP + Eagle 3.1 compatibility patches
# (padded_local_token_to_seq for fp8 DCP prefill, draft model DCP=1 override)
RUN set -eux; \
    python3 /tmp/tinfoil-patches/apply_dcp_eagle_patches.py

# Step 5: Clean up __pycache__
RUN find /usr/local/lib/python3.12/dist-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + || true

# Step 6: Verify imports
RUN python3 -c "import vllm; print('vllm', vllm.__version__, 'with DCP+fp8+Eagle patches')"

# Step 7: Clean up patch files
RUN rm -rf /tmp/tinfoil-patches

# Bake FlashInfer cubins at build time (saves ~16min on first boot)
RUN set -eux; \
    flashinfer show-config; \
    flashinfer download-cubin; \
    cubin_dir=/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins; \
    du -sh "$cubin_dir"; \
    test "$(find "$cubin_dir" -type f | wc -l)" -gt 1000
