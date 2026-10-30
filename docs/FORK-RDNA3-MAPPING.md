# `nicejunjie/Strata-rdna3`: what it changes, and what is worth taking

The fork is at [github.com/nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3), branch `rdna3`, HEAD
`fc8f823` at the time of this reading. It is **not** based on a current upstream: its `CMakeLists.txt` says engine
**0.1.43**, where upstream is 0.1.54 (v0.2.8) and this box runs 0.1.49.2. That matters: a diff against it mixes
upstream's own 0.1.43 → 0.1.54 work with the fork's. Everything below was read from the fork's source, not from its
README.

Its root commit `012ce90` is a squashed import, so most features have no individual commit. The ones that do:
`02602e1` (STRATA_SPEC_CUM), `c3b16b1` (STRATA_PA_KLDS), `84b27e8` (STRATA_GR_FUSED), `d83ecfc` (CPU IQ3_S gate/up
byte planes), `f74ba4f` (wide top-k on HIP - already upstream, see below).

## 1. Switches upstream already has (and leaves off on gfx1100)

Upstream's `src/core/arch_defaults.cpp` turns these on for **gfx1151 (Strix Halo) and gfx12 (RDNA4)** only. A
gfx1100 card runs their default path. That is the cheapest part of the fork's advantage to take, because it is
environment variables, not a patch:

| switch | what it does |
|---|---|
| `STRATA_HIP_WMMA` / `STRATA_SELECT_WMMA` | tensor-core (WMMA) prompt GEMM and QSA selection |
| `STRATA_HC_UPMIX` / `STRATA_PF_HCDOWN` | the hyper-connection up/down passes in the mixed-precision path |
| `STRATA_HEAD_MIX_MULTI` | several heads' mixed projections in one launch |
| `STRATA_GDN_CHUNKED` / `GDN_HEAD` / `GDN_PP` | the GDN (delta) layer's chunked/head/pipelined forms |
| `STRATA_PF_FUSED` / `PF_GEMM` / `PF_F16` / `PF_PAD` | prompt-path GEMM selection, fusion, fp16 staging, padding |
| `STRATA_QFUSE` | fused quantize for the prompt path |
| `STRATA_PA_FAST` | the fast prompt-attention kernel |
| `STRATA_KV_HOST_DMA` | host→device KV staging through DMA |
| `STRATA_VERIFY_QDEDUP` | deduplicated verify-window quantization |
| `STRATA_PREFILL_AUTO_MAX` / `PREFILL_CPU_SHARE` / `STAGER_RING` | how a prompt is split between CPU and GPU experts |

Measured on this box (oct3, one 7900 XTX, `--prefill auto:32768`, `--kv-resident 65536`), turning all of them on at
once:

| | base | with the switches | delta |
|---|---|---|---|
| prefill 32K | 1880 tok/s | 2104 tok/s | +11.9% |
| prefill 96K | 1794 tok/s | 2000 tok/s | +11.5% |
| decode ms per forward, 32K / 96K | 25.00 / 25.51 | 26.08 / 25.23 | within noise |
| 16-prompt decode median | 90.4-90.8 tok/s | 93.85 tok/s | +3.4% |

The exact arm:

```
STRATA_PREFILL_RING=192 STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=6144 \
STRATA_QFUSE=1 STRATA_PA_FAST=1 STRATA_HC_UPMIX=1 STRATA_HEAD_MIX_MULTI=1 \
STRATA_VERIFY_QDEDUP=1 STRATA_KV_HOST_DMA=1 STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 \
STRATA_PF_PAD=1 STRATA_PF_HCDOWN=1
```

**Not deployed.** `PF_GEMM`, `PF_FUSED` and `PA_FAST` change prefill rounding, and this arm was measured on
0.1.49.2 without a KL check. Next step: ablate one switch at a time (prefill speed), then `tools/quant_quality.py`
on the survivors, then deploy.

## 2. What upstream does not have at all

Verified by grep of v0.2.8's `src` and `include`: none of these names occur.

| fork switch / feature | kind | where | fork's own claim |
|---|---|---|---|
| `STRATA_SPEC_CUM` | decode | draft cumulative-probability floor (adapted from android-imdad/Strata-7900xtx, `02602e1`) | fewer wasted verify slots |
| `STRATA_COMBINE_CPU` | decode | the expert combine reads the CPU share's rows straight from the mapped buffer, fused with the GPU rows | one less pass over the CPU share |
| `STRATA_WAIT_FUSE` | decode | the CPU-share wait moved into every combine block | fewer stalls at the combine |
| `STRATA_ROUTE_PUBLISH` | decode | top-10 route and the doorbell publish in one launch | one launch fewer per layer |
| `STRATA_RDNA_MMVQ_WAVE` | decode | gfx1100 wave count for the multi-column GEMV (1 = all shapes, 2 = measured small formats) | small-format GEMVs |
| `STRATA_RDNA_Q6K` | decode | gfx1100 multi-column Q6_K x q8_1 GEMV (`rdna_mmvq.cu`) | Q6_K layers only |
| `STRATA_GR_RDNA*`, `GR_FUSED`, `GR_INJ_UP` | decode | hyper-connection int8 GEMV on RDNA; `GR_FUSED` runs down and up in one launch with a grid barrier (`84b27e8`) | bitwise; IQ3 single 20.69 → 20.54 ms/window, dual suite 136.9 → 139.7 (mixed) |
| `STRATA_PA_KLDS` | prefill | prompt attention stages K through LDS like V (`c3b16b1`) | bitwise; QSA attention 1102 → 993 ms at 32K |
| `STRATA_PREFILL_MMQ_TILE_ROWS` | prefill | tile rows for the prompt MMQ GEMM (`moe_mmq.cu`) | tuning knob |
| CPU IQ3_S / IQ3_XXS gate/up byte planes | both | AVX-512 `vpermt2b` rows for the CPU expert share (`d83ecfc`) | bit for bit ggml-cpu's dot |

`f74ba4f` (wide top-k compiled on HIP) is **not** missing: v0.2.8 has `block_topk_wide_kernel` without a HIP guard.

The fork's posted 32K prefill (3511 tok/s) and decode (100.6 tok/s) on a 7900 XTX with a Zen2 / PCIe 3 host (lcz.me
topic 2671) also depend on the WMMA prompt attention and QSA selection. This repo's CMake probes **rocWMMA** and
finds none, so that path cannot be built here as-is; taking it means adding the rocWMMA headers and a gfx1100 build
of them, which is a separate, larger piece of work.

## 3. Order to port, and how to prove each one

One item per branch, each with the same acceptance gate:

1. build with the item on, the item off, everything else identical;
2. `perf/harness/win_perf.sh` at 32K and 96K, 4 reps, same day (prefill tok/s, ms per forward);
3. `perf/harness/win_tps2.sh` (16 prompts, median tok/s, acceptance);
4. if the item can change arithmetic (everything but `PA_KLDS` and `GR_FUSED`, which say "bitwise"), a
   `tools/quant_quality.py` run against fp16 K/V and the same numbers as `perf/kl-tables.txt`;
5. needles at 96K/128K if the item touches K/V or the selection;
6. only then the production unit, with the previous binary kept for rollback (`docs/DEPLOY.md` §6).

Suggested order, cheapest and most likely first: **`SPEC_CUM`** (self-contained, decode, no arithmetic change to the
forward), then **`COMBINE_CPU` / `WAIT_FUSE` / `ROUTE_PUBLISH`** (three small decode/combine changes), then the **CPU
IQ3_S byte-plane kernel** (this box's `--pool-workers 15` makes the CPU share a real part of every forward), then
**`PA_KLDS` / `PREFILL_MMQ_TILE_ROWS`** (prefill, one of them bitwise), then **`RDNA_MMVQ_WAVE` / `RDNA_Q6K`**, then
**`GR_*`**.

Do not port a switch without its measurement: the fork itself records `PREFILL_RAMP` as "measured no gain" and one
`RDNA_IQ3S` kernel as "correct but slower than the native kernel at T >= 2", which is what a port looks like when the
number was taken before the code was kept.

## 4. What we did not take, and why

- **The fork's tree as a base.** 0.1.43 against 0.1.54 is ~170 commits of upstream work (conversation snapshots,
  park VRAM slots, new prompt paths); starting from the fork would mean re-porting all of it. Everything above is
  small enough to move forward instead.
- **Defaults that change rounding silently.** Each port lands opt-in, like upstream's own new features.
- **Multi-GPU features** the fork carries (`STRATA_DP_FULL_RES`, `MTP_STAGE0`, ...): this box has one card.
