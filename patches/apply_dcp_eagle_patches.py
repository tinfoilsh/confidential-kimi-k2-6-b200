#!/usr/bin/env python3
"""
Patch vLLM v0.23.0 for Kimi K2.6 DCP + Eagle 3.1 + fp8 KV cache.

This script applies three patches on top of PR #40750's partial apply:

1. mla_attention.py: Populate padded_local_token_to_seq for DCP+fp8 prefill path.
   (PR #40750 hunk that failed to apply cleanly against v0.23.0.)

2. llm_base_proposer.py: Override _create_draft_vllm_config to set DCP=1 for
   the Eagle draft model and apply draft_attention_backend/draft_kv_cache_dtype.
   (Based on PR #40611: [SpecDecode] Allow draft-specific attention backend and KV dtype.)

3. speculative.py: Add draft_attention_backend and draft_kv_cache_dtype fields
   to SpeculativeConfig. (From PR #40611, already partially applied via
   apply_manual_patches.py - this is a no-op if already present.)

These patches enable:
  - TRITON_MLA attention backend with DCP=8 and fp8 KV cache
  - Eagle 3.1 speculative decoding with DCP (draft model runs without DCP)
  - RunAI model streamer for faster loading

Usage:
  python3 apply_dcp_eagle_patches.py

Prerequisites:
  - vLLM v0.23.0 installed at /usr/local/lib/python3.12/dist-packages/vllm/
  - PR #40750 patch applied (triton_mla.py replacement, mla_attention.py partial)
  - apply_manual_patches.py already run (assert k_scale removal, speculative.py
    fields, flashinfer.py maybe_override_cp)
"""
import sys


def patch_file(path, checks, patches):
    """Apply patches to a file, skipping if already applied."""
    with open(path) as f:
        src = f.read()

    for check_str, (old, new, desc) in zip(checks, patches):
        if check_str in src:
            print(f"  SKIP: {desc} (already present)")
            continue
        if old not in src:
            print(f"  ERROR: Could not find insertion point for {desc}")
            print(f"  Path: {path}")
            sys.exit(1)
        src = src.replace(old, new, 1)
        print(f"  PATCH: {desc}")

    with open(path, 'w') as f:
        f.write(src)


print("Applying DCP + Eagle 3.1 + fp8 patches...")

# ============================================================================
# Patch 1: mla_attention.py - Populate padded_local_token_to_seq for DCP+fp8
# ============================================================================
print("\n[1/3] Patching mla_attention.py (DCP fp8 prefill path)...")
path_mla = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py'

# 1a: Add computation of padded_local_token_to_seq before _ChunkedMetadata constructor
old_compute = """                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )

                prefill_tokens_with_context = None"""

new_compute = """                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )

                    # Compute padded-local token_to_seq and total_token
                    # for gather_and_maybe_dequant_cache (FP8 DCP support)
                    padded_local_chunk_total_token = (
                        padded_local_cu_chunk_seq_lens_cpu[:, -1]
                    )
                    padded_local_max_token_num = (
                        padded_local_chunk_total_token.max().item()
                    )
                    padded_local_token_to_seq_cpu = torch.zeros(
                        [num_chunks, padded_local_max_token_num],
                        dtype=torch.int32,
                    )
                    for i in range(num_chunks):
                        t2s = torch.repeat_interleave(
                            range_idx, padded_local_chunk_seq_lens[i]
                        )
                        padded_local_token_to_seq_cpu[i, : t2s.shape[0]] = t2s

                prefill_tokens_with_context = None"""

# 1b: Pass padded_local_token_to_seq to ChunkedContextMetadata constructor
old_ctor = """                        prefill_tokens_with_context=prefill_tokens_with_context,
                    )
                else:"""

new_ctor = """                        prefill_tokens_with_context=prefill_tokens_with_context,
                        padded_local_token_to_seq=(
                            padded_local_token_to_seq_cpu.to(device, non_blocking=True)
                        ),
                        padded_local_chunk_total_token=(
                            padded_local_chunk_total_token.tolist()
                        ),
                    )
                else:"""

patch_file(path_mla,
    ['padded_local_token_to_seq_cpu'],
    [(old_compute, new_compute, 'padded_local_token_to_seq computation'),
     (old_ctor, new_ctor, 'padded_local_token_to_seq in constructor')])

# ============================================================================
# Patch 2: llm_base_proposer.py - Override _create_draft_vllm_config for DCP
# ============================================================================
print("\n[2/3] Patching llm_base_proposer.py (draft model DCP override)...")
path_proposer = '/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py'

old_draft = """    def _create_draft_vllm_config(self) -> VllmConfig:
        \"\"\"Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        \"\"\"
        spec_cfg = self.speculative_config
        base = self.vllm_config

        if spec_cfg.moe_backend is not None:
            base = replace(
                base,
                kernel_config=replace(
                    base.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )

        # Note (matt): Never inherit the attention backend from base, because there are
        # many opportunities for incompatibility, so we always independently autoselect
        # unless explicitly specified in the speculative config.
        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        return base"""

new_draft = """    def _create_draft_vllm_config(self) -> VllmConfig:
        \"\"\"Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        \"\"\"
        spec_cfg = self.speculative_config
        base = self.vllm_config

        # Override parallel config: draft model should not use DCP/PCP
        # even if the target model does (PR #40611).
        if spec_cfg.draft_parallel_config is not None:
            base = replace(
                base,
                parallel_config=replace(
                    spec_cfg.draft_parallel_config,
                    rank=base.parallel_config.rank,
                ),
            )
        else:
            # Force DCP=1 and PCP=1 for the draft model
            base = replace(
                base,
                parallel_config=replace(
                    base.parallel_config,
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=1,
                ),
            )

        if spec_cfg.moe_backend is not None:
            base = replace(
                base,
                kernel_config=replace(
                    base.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )

        # Apply draft KV cache dtype override (PR #40611)
        if spec_cfg.draft_kv_cache_dtype is not None:
            base = replace(
                base,
                cache_config=replace(
                    base.cache_config,
                    cache_dtype=spec_cfg.draft_kv_cache_dtype,
                ),
            )

        # Apply draft attention backend override (PR #40611)
        # Never inherit the attention backend from base, because there are
        # many opportunities for incompatibility, so we always independently
        # autoselect unless explicitly specified in the speculative config.
        draft_backend = spec_cfg.draft_attention_backend
        if draft_backend is None:
            draft_backend = spec_cfg.attention_backend
        if draft_backend == "auto":
            draft_backend = None
        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=draft_backend,
            ),
        )

        return base"""

patch_file(path_proposer,
    ['spec_cfg.draft_parallel_config'],
    [(old_draft, new_draft, '_create_draft_vllm_config DCP+draft overrides')])

# ============================================================================
# Patch 3: Verify speculative.py has draft_attention_backend/draft_kv_cache_dtype
# ============================================================================
print("\n[3/3] Verifying speculative.py (draft fields)...")
path_spec = '/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py'
with open(path_spec) as f:
    spec_src = f.read()

if 'draft_kv_cache_dtype' in spec_src and 'draft_attention_backend' in spec_src:
    print("  OK: draft_kv_cache_dtype and draft_attention_backend already present")
else:
    print("  WARNING: draft_kv_cache_dtype/draft_attention_backend not found!")
    print("  Run apply_manual_patches.py first to add these fields.")

print("\nAll patches applied successfully!")
