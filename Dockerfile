# syntax=docker/dockerfile:1.6
#
# Confidential Kimi K2.6 with DCP + Eagle 3.1 + fp8 KV cache.
# Base is digest-pinned for attestation.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.23.0-ubuntu2404@sha256:662e4975c5c9947f8723f4d8f438145971361a480a2ade1919bb9462a9f24088
FROM ${VLLM_BASE_IMAGE}

# Patches are -p1 unified diffs rooted at /; they target
# usr/local/lib/python3.12/dist-packages/... to match the base image.
# 0001 is a fused backport of vllm PRs #40750 + #40609 + #40611, rewritten
# to apply cleanly against v0.23.0 (no --forward, no .rej cleanup needed).
# It supersedes the previous multi-patch stack (0001+0002+0003+0005) and
# includes the fix for the DCP+fp8 crash (padded_local_token_to_seq).
# 0002 patches flashinfer's ensure_symlink() to tolerate read-only rootfs.
# triton_mla.py and triton_mla_tuning.py are full file replacements from PR #40750.
COPY patches/ /tmp/tinfoil-patches/
RUN set -eux; \
    cd /; \
    patch -p1 --no-backup-if-mismatch --fuzz=0 \
        < /tmp/tinfoil-patches/0001-dcp-fp8-triton-mla-eagle-v0230.patch; \
    patch -p1 --no-backup-if-mismatch --fuzz=0 \
        < /tmp/tinfoil-patches/0002-flashinfer-ro-symlinks.patch; \
    cp /tmp/tinfoil-patches/triton_mla.py \
        /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla.py; \
    cp /tmp/tinfoil-patches/triton_mla_tuning.py \
        /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla_tuning.py; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + || true; \
    rm -rf /tmp/tinfoil-patches; \
    python3 -c "import vllm; print('vllm', vllm.__version__, 'with DCP+fp8+Eagle patches')"

# Bake FlashInfer cubins at build time (saves ~16min on first boot).
# Pre-create the symlinks that flashinfer's JIT creates at runtime via
# ensure_symlink() — on a read-only container rootfs the runtime mkdir +
# lockfile creation fails.  The symlink paths must match what the JIT
# code expects (see flashinfer/jit/fused_moe.py and gemm/core.py):
#   flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export
#   flashinfer/trtllm/gemm/trtllmGen_gemm_export
# 0004 also patches ensure_symlink() itself as a belt-and-suspenders
# fallback in case a symlink is missing or stale.
RUN set -eux; \
    flashinfer show-config; \
    flashinfer download-cubin; \
    cubin_dir=/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins; \
    du -sh "$cubin_dir"; \
    test "$(find "$cubin_dir" -type f | wc -l)" -gt 1000; \
    mkdir -p "$cubin_dir/flashinfer/trtllm/batched_gemm" "$cubin_dir/flashinfer/trtllm/gemm"; \
    for d in "$cubin_dir"/*/; do \
        gemm_dir=$(find "$d" -maxdepth 3 -type d -name "trtllmGen_gemm_export" 2>/dev/null | head -1); \
        bmm_dir=$(find "$d" -maxdepth 3 -type d -name "trtllmGen_bmm_export" 2>/dev/null | head -1); \
        if [ -n "$gemm_dir" ]; then \
            ln -sf "$gemm_dir" "$cubin_dir/flashinfer/trtllm/gemm/trtllmGen_gemm_export"; \
        fi; \
        if [ -n "$bmm_dir" ]; then \
            ln -sf "$bmm_dir" "$cubin_dir/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"; \
        fi; \
    done; \
    ls -la "$cubin_dir/flashinfer/trtllm/batched_gemm/" "$cubin_dir/flashinfer/trtllm/gemm/"; \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__, 'cubins baked')"
