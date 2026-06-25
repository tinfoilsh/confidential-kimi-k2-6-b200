# Confidential Kimi K2.6 - DCP + Eagle 3.1 + fp8 KV cache

Patched vLLM v0.23.0 image for Kimi K2.6 on Blackwell (B200/B300) with:
- **Decode Context Parallel (DCP=8)**: ~16x KV cache capacity (39.9M tokens vs 2.4M)
- **TRITON_MLA attention backend** with fp8 KV cache
- **Eagle 3.1 speculative decoding** (draft model runs without DCP)
- **RunAI model streamer** for faster weight loading

## Patches

Based on upstream vLLM PRs (not yet merged as of v0.23.0):
- **PR #40750**: TRITON_MLA MTP full CUDA graphs for Kimi on Blackwell (DCP+fp8, tuning table)
- **PR #40609**: Enable FP8 KV cache with DCP for MLA
- **PR #40611**: Allow draft-specific attention backend and KV dtype

### Files
| File | Description |
|------|-------------|
| `Dockerfile` | Builds from `vllm/vllm-openai:v0.23.0-ubuntu2404`, applies all patches, bakes FlashInfer cubins |
| `patches/0001-pr40750-dcp-fp8-triton-mla.patch` | PR #40750 diff adapted to dist-packages paths |
| `patches/triton_mla.py` | Full replacement from PR #40750 (DCP metadata builder, CG-safe buffers) |
| `patches/triton_mla_tuning.py` | New tuning table from PR #40750 (SM120/fp8 kernel configs) |
| `patches/apply_manual_patches.py` | Manual patches for hunks that failed to apply: assert k_scale removal, speculative.py draft fields, flashinfer.py maybe_override_cp |
| `patches/apply_dcp_eagle_patches.py` | DCP+fp8 prefill path (padded_local_token_to_seq) + Eagle draft model DCP=1 override |
| `tinfoil-config.yml` | Tinfoil deployment config with DCP=8, TRITON_MLA, fp8, Eagle 3.1, RunAI streamer |

## Build

```bash
docker build -t ghcr.io/tinfoilsh/confidential-kimi-k2-6-b200-dcp:latest .
```

## Launch config

Key vLLM arguments:
```
--attention-backend TRITON_MLA
--decode-context-parallel-size 8
--kv-cache-dtype fp8
--tensor-parallel-size 8
--load-format runai_streamer
--speculative-config '{"model":"...","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8"}'
```

Environment variables:
- `RUNAI_STREAMER_MEMORY_LIMIT=4294967296` (4GB streamer buffer)
- `RUNAI_STREAMER_CONCURRENCY=8` (8 concurrent streams)
- `FLASHINFER_NO_DOWNLOAD=1` (cubins baked at build time)

## Performance

Stress test: 48 concurrent requests, 217K-token unique prompts, 8000 max_tokens:

| Metric | Without DCP | With DCP=8 |
|--------|------------|-----------|
| KV cache blocks | 152,840 | 311,594 |
| KV cache tokens | ~2.4M | ~39.9M |
| Peak KV usage (48x217K) | 96-98% | 1.5% |
| Preemptions | Frequent | 0 |
| Crashes | Engine stalls at >97% | None |
