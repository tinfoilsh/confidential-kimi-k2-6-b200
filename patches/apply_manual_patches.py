#!/usr/bin/env python3
"""Manually patch v0.23.0 files for DCP+fp8 support (PR #40750 hunks that failed to apply)."""
import sys

# Step 3: Patch mla_attention.py - only the assert k_scale removal remains
# (hunks 7-8 already applied the fp8 gather path, ChunkedContextMetadata fields,
#  fp8_ds_mla guard, and k_scale=k_scale call)
path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py'
with open(path) as f:
    src = f.read()

# Remove 'assert k_scale is None' in _context_parallel_compute_prefill_context
old = '        assert k_scale is None, "DCP not support scaled kvcache now."'
new = '        # DCP+fp8: k_scale is now supported (PR #40750)'
if old in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print(f"Patched {path}: removed assert k_scale is None")
else:
    print(f"SKIP {path}: assert k_scale is None already removed or not found")

# Step 4: Patch speculative.py for draft_attention_backend/draft_kv_cache_dtype
path = '/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py'
with open(path) as f:
    src = f.read()

changed = False

# 4a. Add imports
old_import = 'from vllm.config import LoadConfig'
new_import = 'from vllm.config import LoadConfig\nfrom vllm.config.cache import CacheDType'
if old_import in src and 'from vllm.config.cache import CacheDType' not in src:
    src = src.replace(old_import, new_import, 1)
    changed = True

old_import2 = 'from vllm.utils.hashing import safe_hash'
new_import2 = (
    'from vllm.utils.hashing import safe_hash\n'
    'from vllm.v1.attention.backends.registry import AttentionBackendEnum'
)
if old_import2 in src and 'AttentionBackendEnum' not in src:
    src = src.replace(old_import2, new_import2, 1)
    changed = True

# 4b. Add field_validator import if not present
if 'field_validator' not in src:
    src = src.replace(
        'from pydantic import Field, SkipValidation, model_validator',
        'from pydantic import Field, SkipValidation, field_validator, model_validator'
    )
    changed = True

# 4c. Add draft_kv_cache_dtype and draft_attention_backend fields
old_field = '    max_model_len: int | None = Field(default=None, ge=1)'
new_fields = (
    '    draft_kv_cache_dtype: CacheDType | None = None\n'
    '    """KV cache dtype to use for the draft model. When None, the draft\n'
    '    model inherits the target model\'s --kv-cache-dtype setting."""\n'
    '    draft_attention_backend: AttentionBackendEnum | Literal["auto"] | None = None\n'
    '    """Attention backend to use for the draft model. When None, the draft\n'
    '    model inherits the target model\'s attention backend."""\n'
    '    max_model_len: int | None = Field(default=None, ge=1)'
)
if old_field in src and 'draft_kv_cache_dtype' not in src:
    src = src.replace(old_field, new_fields, 1)
    changed = True

# 4d. Add field_validator for draft_attention_backend
old_compute = '    def compute_hash(self) -> str:'
new_validator = (
    '    @field_validator("draft_attention_backend", mode="before")\n'
    '    @classmethod\n'
    '    def validate_draft_attention_backend_before(cls, value):\n'
    '        if isinstance(value, str):\n'
    '            if value.lower() == "auto":\n'
    '                return "auto"\n'
    '            return AttentionBackendEnum[value.upper()]\n'
    '        return value\n\n'
    '    def compute_hash(self) -> str:'
)
if old_compute in src and 'validate_draft_attention_backend_before' not in src:
    src = src.replace(old_compute, new_validator, 1)
    changed = True

if changed:
    with open(path, 'w') as f:
        f.write(src)
    print(f"Patched {path}")
else:
    print(f"SKIP {path}: already patched or patterns not found")

# Step 5: Patch flashinfer.py for maybe_override_cp_for_vllm_config
path = '/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py'
with open(path) as f:
    src = f.read()

old = (
    '        self.use_dcp = self.dcp_world_size > 1\n'
    '        self.dcp_a2a = (\n'
    '            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"\n'
    '        )'
)
new = (
    '        self.maybe_override_cp_for_vllm_config(vllm_config)\n'
    '        self.use_dcp = self.dcp_world_size > 1\n'
    '        self.dcp_a2a = (\n'
    '            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"\n'
    '        )'
)
if old in src and 'maybe_override_cp_for_vllm_config' not in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print(f"Patched {path}")
else:
    print(f"SKIP {path}: already patched or pattern not found")

print("All manual patches applied successfully!")
