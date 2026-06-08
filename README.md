# confidential-kimi-k2-6-b200

Patched vLLM v0.22.0 image for Kimi K2.6 NVFP4 on Blackwell (B200/B300) with Eagle 3.1 speculative decoding.

> **v0.22.1 note:** v0.22.1 currently crashes during Kimi load with a CUTLASS DSL ICE (`mlir_global_dtors`). We stay on v0.22.0 + the fc_norm patch until that is fixed upstream.

## Image

- Base: `vllm/vllm-openai:v0.22.0` (digest-pinned in `Dockerfile`)
- Patches: Eagle 3.1 `fc_norm` support (`patches/0001-eagle31-deepseek-fc-norm.patch`, cherry-pick of vllm#43482)
- FlashInfer cubins: baked at build time via `flashinfer download-cubin`

## FlashInfer cubins (B200 vs B300)

Both B200 (SM100 / `sm_100a`) and B300 (SM103 / `sm_103a`) use the same FlashInfer NVFP4 + TRTLLM-gen kernel path for this config (`tokenspeed_mla`, `VLLM_USE_FLASHINFER_MOE_FP4=1`).

`flashinfer download-cubin` downloads **all** published Blackwell-family artifacts in one shot:

- Shared TRTLLM-gen FMHA/BMM/GEMM cubins (used on all Blackwell GPUs)
- DeepGEMM cubins
- CuteDSL FMHA cubins for `sm_100a`, `sm_103a`, and `sm_110a`

You do **not** need separate images for B200 vs B300. Runtime picks the matching arch automatically from the GPU capability.

Build-time egress to `edge.urm.nvidia.com` is required (CI has open internet). The workload container has **no egress network** — cubins must be baked into the image. At runtime, `FLASHINFER_NO_DOWNLOAD=1` prevents any download attempts.

Do **not** mount a tmpfs over `flashinfer_cubin/cubins` (the whole tree); that would hide the baked cubins. FlashInfer writes runtime symlinks and `.lock` files under `cubins/flashinfer/` only — mount a **narrow** tmpfs there (see `tinfoil-config.yml`).

## Draft model

Mount `lightseekorg/kimi-k2.6-eagle3.1-mla` into the container at `/tinfoil/models/kimi-k2.6-eagle3.1-mla` (host volume or MPK). Example host bind:

```yaml
volumes:
  - /path/on/host/kimi-k2.6-eagle3.1-mla:/tinfoil/models/kimi-k2.6-eagle3.1-mla:ro
```

## Build

```bash
docker build --network host -t confidential-kimi-k2-6-b200 .
```

The cubin download step adds ~1.5 GB to the image and takes several minutes.
