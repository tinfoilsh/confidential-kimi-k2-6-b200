# confidential-kimi-k2-6-b200

Patched vLLM v0.25.0 image for Kimi K2.6 NVFP4 on Blackwell (B200/B300).

## Image

- Base: `vllm/vllm-openai:v0.25.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/triton_mla.py`, `patches/triton_mla_tuning.py` — full-file
    replacements derived from
    [vllm#40750](https://github.com/vllm-project/vllm/pull/40750) (TRITON_MLA
    spec-as-decode + full CUDA graphs for MTP/eagle), rebased onto v0.25.0 so
    they carry upstream's stride-aware kernel signature (#45111) and
    `get_kv_cache_stride_order`; retire when #40750 lands in a release we
    consume
  - `patches/0001-spec-decode-proposer-event-race.patch` — re-records the
    prepare-inputs event after the spec-decode proposer so the next batch
    cannot mutate block tables the proposer is still reading (part of the
    #40750 stack, still unfixed upstream)
  - `patches/0002-flashinfer-ro-symlinks.patch` — flashinfer `ensure_symlink()`
    read-only rootfs fix
- Production config: `--tensor-parallel-size 8`,
  `--decode-context-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--enable-expert-parallel`, `--attention-backend TRITON_MLA`,
  `--moe-backend flashinfer_trtllm`,
  `--speculative-config '{"method":"eagle3","num_speculative_tokens":1,"attention_backend":"TRITON_MLA",...}'`
- `num_speculative_tokens` is pinned to 1: vLLM's eagle multi-step drafting
  loop is not DCP-aware (per-step slot mapping uses global positions and
  `dcp_local_seq_lens` is never recomputed inside the loop), so under DCP=8
  draft tokens beyond the first are generated from corrupted draft KV and
  their acceptance collapses. Raise it only after validating acceptance on
  hardware or after upstream makes the drafting loop DCP-aware.
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`
- FlashInfer cubins baked at build time (`FLASHINFER_NO_DOWNLOAD=1`);
  `NUMBA_CACHE_DIR` points into tmpfs for the fused Kimi image-preprocessing
  JIT cache (rootfs is read-only)

## Build

```bash
docker build -t ghcr.io/tinfoilsh/confidential-kimi-k2-6-b200 .
```
