# syntax=docker/dockerfile:1.6
#
# Patched vLLM image. Base is digest-pinned for attestation. See patches/
# for the diff set and README.md for the patching playbook.
#
# Note: v0.22.1 hits a CUTLASS DSL ICE (mlir_global_dtors) during Kimi K2.6
# model load on Blackwell; stay on v0.22.0 until upstream fixes land.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.22.0@sha256:0fec7ec5f3e6bc168e54899935fb0557da908a4832a1dbc88e2debcf2f889416
FROM ${VLLM_BASE_IMAGE}

# Patches are -p1 unified diffs rooted at /; they target
# usr/local/lib/python3.12/dist-packages/... to match the base image.
COPY patches/ /tmp/tinfoil-patches/
RUN set -eux; \
    test -x /usr/bin/patch; \
    cd /; \
    for p in /tmp/tinfoil-patches/*.patch; do \
        echo "Applying $(basename "$p")"; \
        /usr/bin/patch -p1 --no-backup-if-mismatch --fuzz=0 < "$p"; \
    done; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + || true; \
    rm -rf /tmp/tinfoil-patches; \
    python3 -c "import vllm; print('vllm', vllm.__version__, 'with tinfoil patches')"

# Bake FlashInfer cubins at build time. CI needs egress to edge.urm.nvidia.com;
# the workload container runs with FLASHINFER_NO_DOWNLOAD=1 and no network.
RUN set -eux; \
    flashinfer show-config; \
    flashinfer download-cubin; \
    cubin_dir=/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins; \
    du -sh "$cubin_dir"; \
    test "$(find "$cubin_dir" -type f | wc -l)" -gt 1000
