# confidential-kimi-k2-6-b200

Patched vLLM v0.23.0 image for Kimi K2.6 on Blackwell (B200/B300).

## Image

- Base: `vllm/vllm-openai:v0.23.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/0001-dcp-fp8-triton-mla-eagle-v0230.patch` — fused backport of
    [vllm#40750](https://github.com/vllm-project/vllm/pull/40750),
    [vllm#40609](https://github.com/vllm-project/vllm/pull/40609), and
    [vllm#40611](https://github.com/vllm-project/vllm/pull/40611) for v0.23.0.
    All hunks rewritten to match v0.23.0's code structure (no `--forward`,
    no `.rej` cleanup). Includes the fix for the DCP+fp8 crash where
    `padded_local_token_to_seq` was not computed (assertion failure under
    concurrent prefill with `--kv-cache-dtype fp8` + DCP > 1).
  - `patches/0002-flashinfer-ro-symlinks.patch` — flashinfer `ensure_symlink()`
    read-only rootfs fix
  - `patches/triton_mla.py`, `patches/triton_mla_tuning.py` — full file replacements from PR #40750
- Production config: `--tensor-parallel-size 8`,
  `--decode-context-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--attention-backend TRITON_MLA`,
  `--speculative-config '{"method":"eagle3","num_speculative_tokens":3,...}'`
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`
- FlashInfer cubins baked at build time (`FLASHINFER_NO_DOWNLOAD=1`)

## Build

```bash
docker build -t ghcr.io/tinfoilsh/confidential-kimi-k2-6-b200 .
```
