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

**Deployed on 2026-10-30** as the production unit's environment (engine 0.1.54 from this repo), with one addition the
forum's list has and this engine implements differently: `STRATA_HIP_WMMA` works on gfx1100 (the prompt attention on
matrix cores), while `STRATA_SELECT_WMMA` (the QSA block scorer on matrix cores) in v0.2.8 is gfx12-only, so that half
of the fork's WMMA advantage is not available here.

Measured on this box, oct3, production config, 4 reps (`perf/raw/perf-fa54-*.jsonl`):

| arm | 32K prefill | 96K prefill |
|---|---|---|
| base | 1864 | 1787 |
| the ten arithmetic switches + the staging ring | 2108 | 2008 |
| + `HIP_WMMA` | 2090 | 1928 |
| + GDN / `PF_F16` / `PREFILL_AUTO_MAX` / `STAGER_RING` | **2243** | 1979 |

So the forum's 2x is, on this engine, +13% from the prompt fusion/GEMM/CPU-share group and another +6% from the GDN
chunked recurrence and the staging/chunk group; the matrix-core prompt attention adds nothing measurable on this model
(it does engage - `strata: STRATA_HIP_WMMA: the prompt attention on matrix cores on (gfx1100)`). The rest of their
2x sits in code that v0.2.8 does not have: `SELECT_WMMA` on gfx11, `PA_KLDS`, `PREFILL_MMQ_TILE_ROWS`,
`PF_GROUP_RES`, and the `--ple-io direct` + row-cache pairing.

Quality (`perf/knob-quality.txt`): the stdin `run` path is bitwise identical with and without the set; the batched
prompt path keeps top-1 on all 20 prompts with mean KL 0.0138 (13-switch half) and 0.0170 (full set) against the
unswitched run (median 0.0005, max 0.125). For scale: oct3 with its window is 0.0185 on the KV yardstick.

One crash: a run with `--stats` stalled at the first prompt chunk's layer 0 and the engine's watchdog aborted it 60 s
later (ABRT from `strata-watchdog`; `no progress for 60 s ... stopping the engine so the server starts it again`).
Not reproduced in four other runs of the same switches without `--stats`, including the 16-prompt harness that the
deployed unit passes with 0 stalls. Deployed without `--stats`, with `LimitCORE=0` in the unit so no core is written.

## 2. What upstream does not have, and its status in v0.2.8

Verified by grep of v0.2.8's `src` and `include`. "Absent" means the behaviour is not there under any name I could
find; "in another form" names what is there instead.

| fork switch / feature | kind | fork's own claim | status in v0.2.8 |
|---|---|---|---|
| `STRATA_SPEC_CUM` | decode | draft cumulative-probability floor (android-imdad/Strata-7900xtx, `02602e1`) | **in another form, and only for DFlash2**: `--draft-conf` gates the chain product in the `dflash2` drafter; the MTP window (`--mtp`, this box) has only `--spec-min-p`'s per-draft floor. **Ported for MTP in this fork** (`fork-accel`), see §2a |
| `STRATA_COMBINE_CPU` | decode | the expert combine reads the CPU share's rows from the mapped buffer, fused with the GPU rows | absent. Upstream's closest is `STRATA_LFUSE` (S26: fewer launches around the shared expert and the KV append), which is a different fusion |
| `STRATA_WAIT_FUSE` | decode | the CPU-share wait in every combine block | absent |
| `STRATA_ROUTE_PUBLISH` | decode | top-10 route and doorbell publish in one launch | **in another form**: upstream's `doorbell_publish_res` publishes the resident plan's results (route included) in one call; the route itself is still its own `resident_plan` launch in both |
| `STRATA_RDNA_MMVQ_WAVE` | decode | gfx1100 wave count for the multi-column GEMV | absent. The gfx1100 GEMV table upstream has is the one this fork added in `gfx1100-mmvf-rows1` (rows per block), not a wave switch |
| `STRATA_RDNA_Q6K` | decode | gfx1100 multi-column Q6_K x q8_1 GEMV (`rdna_mmvq.cu`) | absent: v0.2.8 has no `rdna_mmvq.cu`; Q6_K goes through `iq_kernels.cu`'s generic chain |
| `STRATA_PREFILL_MMQ_TILE_ROWS` | prefill | tile rows for the prompt MMQ GEMM (`moe_mmq.cu`) | absent as a knob. `moe_fused_iq.cu` computes its tile rows at compile time (`tile_rows(ww, nt)`), a different kernel |
| `STRATA_GR_RDNA*`, `GR_FUSED`, `GR_INJ_UP` | decode | hyper-connection int8 GEMV on RDNA; down and up in one launch with a grid barrier (`84b27e8`) | absent (`rdna_gr.cu` has no upstream counterpart). Upstream's nearest is `STRATA_GR_V3`, the stream-split read in `fused_gr.cu` |
| `STRATA_PA_KLDS` | prefill | prompt attention stages K through LDS like V (`c3b16b1`, bitwise) | absent, **but the premise moved**: v0.2.8's prompt kernel fetches a chunk's K fragments and scales into a `Chunk` structure ahead of the WMMA loop, which 0.1.43 did not. A port needs a re-measurement, not a mechanical move |
| CPU IQ3_S / IQ3_XXS gate/up byte planes | both | AVX-512 `vpermt2b` rows for the CPU expert share (`d83ecfc`, bit for bit) | absent: `src/kernels/cpu/iq_avx512.cpp` handles IQ3_S row-wise, and has no byte-plane path |

`f74ba4f` (wide top-k compiled on HIP) is **not** missing: v0.2.8 has `block_topk_wide_kernel` without a HIP guard.

### 2a. `STRATA_SPEC_CUM`, ported for the MTP path

`src/program/generate.cpp` only: a `spec_cum_floor()` helper (`STRATA_SPEC_CUM=P`, unset or 0 = off) and the product
of the draft probabilities inside the two MTP window rules (`t_rule` for the serial loop's B chain, and the
`use_mtp && req_spec_min_p > 0` rule for the window). Where the running product falls below `P` the window ends, so
fewer low-value drafts are verified. It changes which drafts are verified, never the sample.

Measured on this box, oct3, production config (`--spec 5 --spec-min-p 0.5`), 16-prompt harness, median over 16
prompts (`perf/raw/tps2-facc-*.txt`):

| arm | tok/s | ms per forward | draft acceptance |
|---|---|---|---|
| base (`STRATA_SPEC_CUM` off) | 90.00 / 90.65 | 30.32 / 30.42 | 60.0% / 60.8% |
| `STRATA_SPEC_CUM=0.25` | 92.15 | 30.64 | 61.8% |
| **`STRATA_SPEC_CUM=0.4`** | **93.05 / 93.55** | 30.49 / 30.36 | 66.3% / 66.3% |
| `STRATA_SPEC_CUM=0.6` | 90.00 | 28.01 | 72.8% |

Two runs at 0.4 against two base runs: **+2.4% to +3.6% tok/s**, the same cost per forward, acceptance up ~6 points.
0.6 cuts the windows too short - the highest acceptance of the four, and no throughput left. 0.4 is what the fork's
`fork-accel` branch recommends; it stays opt-in (an env var, no CLI change).



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
