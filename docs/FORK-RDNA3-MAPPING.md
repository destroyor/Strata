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

**Deployed on 2026-10-30** as the production unit's environment (engine 0.1.54 from this repo). Both WMMA halves work
on gfx1100 after all: `sel_gfx12_device()` in `qsa_select.cu` already accepts gfx11 through
`gfx_arch_is_gfx11_wmma()` (the kernel has a gfx11 fragment layout, `STRATA_SEL_GFX11`), it was only the
`STRATA_SELECT_WMMA` opt-in that was missing from the arm. Measured on this box, oct3, production config, 4 reps
(`perf/raw/perf-fa54-*.jsonl`):

| arm | 32K prefill | 96K prefill |
|---|---|---|
| base | 1864 | 1787 |
| the ten arithmetic switches + the staging ring | 2108 | 2008 |
| + `HIP_WMMA` | 2090 | 1928 |
| + GDN / `PF_F16` / `PREFILL_AUTO_MAX` / `STAGER_RING` | 2243 | 1979 |
| + `SELECT_WMMA` | **2280** | **2304** |

So the forum's 2x is, on this engine, +13% from the prompt fusion/GEMM/CPU-share group, another +6% from the GDN
chunked recurrence and the staging/chunk group, and +16% at 96K from the matrix-core block scorer (the scorer's work
grows with the context, so 32K barely moves). `HIP_WMMA` alone adds nothing measurable on this model (it does engage -
`strata: STRATA_HIP_WMMA: the prompt attention on matrix cores on (gfx1100)`). What is still absent is code v0.2.8 does
not have: `PA_KLDS`, `PREFILL_MMQ_TILE_ROWS`, `PF_GROUP_RES`, and the `--ple-io direct` + row-cache pairing.

Selection quality: `SELECT_WMMA`'s ids are identical to the warp scorer's here - 29,995 positions over five
6,000-token segments, top-1 100.00%, KL 0.00000, full switch set on both sides.

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
| `STRATA_COMBINE_CPU` | decode | the expert combine reads the CPU share's rows from the mapped buffer, fused with the GPU rows | absent. Upstream's closest is `STRATA_LFUSE` (S26: fewer launches around the shared expert and the KV append), which is a different fusion. **Ported in this fork** (`STRATA_COMBINE_CPU`), see §2b |
| `STRATA_WAIT_FUSE` | decode | the CPU-share wait in every combine block | **in another form**: upstream's 143-amdk `STRATA_FUSE_WAIT` folds the wait into the copy that follows it (`wait_copy_i32_from_mapped`, `copy_rows_from_mapped_after`). **Plan-only on HIP now too** (`STRATA_FUSE_WAIT=2`), which is what the fork's switch does; see §2b |
| `STRATA_ROUTE_PUBLISH` | decode | top-10 route and doorbell publish in one launch | **in another form**: upstream's `doorbell_publish_res` publishes the resident plan's results (route included) in one call; the route itself is still its own `resident_plan` launch in both. **Ported in this fork** (`STRATA_ROUTE_PUBLISH`), see §2b |
| `STRATA_RDNA_MMVQ_WAVE` | decode | gfx1100 wave count for the multi-column GEMV | absent. The gfx1100 GEMV table upstream has is the one this fork added in `gfx1100-mmvf-rows1` (rows per block), not a wave switch |
| `STRATA_RDNA_Q6K` | decode | gfx1100 multi-column Q6_K x q8_1 GEMV (`rdna_mmvq.cu`) | absent as this kernel: v0.2.8's `native_mmvq.cu` has Q6_K multi-column through ggml's chain (2 weights per lane per step) and a wave-64 layout. **Ported in this fork and deployed**, see §2c |
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

### 2b. `STRATA_ROUTE_PUBLISH` / `STRATA_COMBINE_CPU` / `STRATA_WAIT_FUSE`, ported

The three decode-path fusions are in this fork, each behind its own env var and **each off by default**, so the
default path is unchanged:

| switch | what it replaces | launch saved per MoE layer |
|---|---|---|
| `STRATA_ROUTE_PUBLISH=1` | `native_router_top10_multi` + `doorbell_publish_res` / `doorbell_publish` | 1 |
| `STRATA_COMBINE_CPU=1` | `wait_flag_ge` + `copy_rows_from_mapped` + `moe_hit_add` + the parts-array combine | 3 |
| `STRATA_FUSE_WAIT=2` | the plan's `wait_flag_ge` + `copy_i32_from_mapped` (the fork's `STRATA_WAIT_FUSE`) | 1 |

(`STRATA_FUSE_WAIT=1` is upstream's older, wider form: it folds the CPU-rows wait into
`copy_rows_from_mapped_after` as well. On HIP `=2` used to mean `=1`; it now means the plan only, as on CUDA.)

Ports: `route_row` split out of `route_multi` plus `route_publish_res` / `route_publish_all`
(`native_router.cu`), and `combine_cpu_k10_vec4` / `native_moe_combine_multi_cpu` (`native_moe.cu`), wired in
`verify.cpp` under the same guards the fork uses (`!ar_on() && !device_plan_ && !remote_opt_`, `k = 10`, 2-8 rows).
Both are bitwise the launches they replace. The fork's remaining half of `COMBINE_CPU` - the shared expert's gate row
riding in the router GEMV (`STRATA_MMVF_AUX`, an aux block in the *multi* BF16 kernel) - was **not** taken: it only
pays off together with `STRATA_LFUSE`, which this box does not run.

Measured on this box (oct3, production config and its 23 switches, engine 0.1.54, binary `40a61eaa`), 16-prompt
harness, one run per arm, median over 16 prompts (`perf/raw/tps2-b-c*.txt`):

| arm | tok/s | ms per forward | acceptance |
|---|---|---|---|
| control | 96.05 / 98.20 | 26.97 / 27.04 | 73.0% / 74.2% |
| `ROUTE_PUBLISH=1` | 97.55 | 27.20 | 72.9% |
| `COMBINE_CPU=1` | 97.85 | 27.10 | 72.9% |
| `COMBINE_CPU=1 FUSE_WAIT=1` | 97.50 / 96.50 / 97.30 | 27.10 / 26.67 / 26.90 | 72.5% / 69.8% / 72.0% |
| `FUSE_WAIT=1` (previous binary) | 96.75 | 27.23 | 72.9% |

Run-to-run spread on this harness is ±1.5% (the two control runs differ by 2.2% by themselves), and the paired
per-prompt comparison puts the three arms between -1.5% and +1.1% tok/s against the control, with ms per forward
0.2-2.2% lower for `COMBINE_CPU`. **Nothing here clears the 2% gate**: three launches of a ~40-layer window are
~0.3 ms of a 27 ms forward at best, and this host does not turn that into visible throughput. They stay in the tree
opt-in and **off**; the production unit's environment is unchanged.

Quality: greedy answers on the harness's three questions are character-identical with all three switches on and off,
and the 16-prompt harness's acceptance is unchanged (72.9% against 73.0%), which is what a bitwise port should show.

### 2c. `STRATA_RDNA_Q6K` / the K-quant multi-column GEMV, ported and deployed

`src/kernels/cuda/rdna_mmvq.cu` (the fork's file, unchanged): one wave32 per output row; for Q6_K every lane takes
8 consecutive weights of a superblock (ql nibble | qh 2 bits, -32), decodes them once and applies them to every
column of the window with two `sudot4` - against ggml's chain, which walks 2 weights per lane per step. The same
file carries Q4_K / Q5_K / IQ4_XS (and IQ3_S) at 8 weights per lane; Q4_K/Q5_K take it from 4 columns, the others
always. `native_mmvq()` asks it first for types 12/13/14/23 when `STRATA_RDNA_Q6K=1`; the fallback is the wave-64
kernel this repo already had. It is **not bitwise** the reference kernel (own summation order, same products).

This model's dense side is exactly what it targets (`grep` of the GGUF: 89 Q6_K tensors - attn q/k/v/gate, qkv,
`output`, the shared expert's gate/up), so decode is where it shows. 16-prompt harness, production env and its 23
switches, binary `f88e46f6`, two runs per arm (`perf/raw/tps2-b-q6k*.txt`):

| arm | tok/s | ms per forward | acceptance |
|---|---|---|---|
| off | 95.70 / 94.25 | 27.28 / 27.76 | 70.7% / 71.2% |
| **`STRATA_RDNA_Q6K=1`** | **97.40 / 99.15** | **26.54 / 26.22** | 71.8% / 71.4% |

**+3.5% tok/s and -4.1% ms per forward**, both runs of the arm above both runs of the control. Answers on the
harness's three QA questions are character-identical, and the 32K needle is 3 of 3 with it on. **Deployed** on the
box on 2026-10-31 (unit environment, 30 switches).

### 2d. What did not move the number, with the numbers

Measured the same way (16-prompt harness, production config) while looking for the fork's 106 tok/s median:

| arm | tok/s | verdict |
|---|---|---|
| control (same binary) | 95.7 / 94.25 | - |
| `STRATA_GDN_PP=1` (the fork's value; we run 2) | 95.4 | neutral |
| `STRATA_SH_STREAM=0 GPU_MAX_HW_QUEUES=6` (both the fork's) | 95.65 | neutral |
| `--pool-workers 8` / `4` (we run 15) | 94.8 / 91.85 | the CPU share is hidden: 4 workers cost 4% |
| `STRATA_SPEC_COST=1` + `--spec 6` | 94.55 | +6-7% on the fork's own protocol, neutral here |
| `--spec 8` + `STRATA_SPEC_COST=1` | 46-51 (their protocol) | the cost model picks wide windows the card cannot pay for |
| `--kv-resident 262144` (all cells resident) | 89.4 / 85.7 (theirs protocol) | worse than 65536 |

And the fork's whole argument list on our engine (`--kv int8 --kv-resident 32768 --ple-io direct --ple-row-cache
1048576 --mtp-window 4096 --mtp-q4 all --expert-cache auto`, their env): 94.7 / 87.5 tok/s on their protocol at
32K / 64K against our 89.9 / 80.8 - real (+5% / +8%), but its `--expert-cache auto` computes only 6773 slots here
(17.76 GiB free, 512 MiB reserved) where our fixed 7700 gives 8854, and it leaves 502 MiB of VRAM free with the
engine's own warning attached.

Where the rest of their lead is: on their protocol our cost per forward is already at parity (28.3 ms against their
28.2 at 32K), and the gap is **drafts accepted per forward** (1.56 / 1.29 against their 1.84 / 1.91 at 32K / 64K,
i.e. 70% / 63% per-draft acceptance against 78% / 82%). Swapping the document for verbatim-repeated or random-words
text moves our acceptance only between 60% and 67%, so it is not our text being harder - their drafter's proposals
land better on their engine (0.1.43) than ours do on 0.1.54, with the same draft head file.

### 2e. The window cut (`--spec-min-p`), scanned

The floor decides how many of the drafter's proposals the window verifies. It only chooses which rows are verified,
never the sample, so it is the one remaining throughput knob that costs no quality. Scan on the deployed binary
(16-prompt harness, production config and env):

| `--spec-min-p` | tok/s | ms per forward | acceptance |
|---|---|---|---|
| 0.5 (deployed) | 96.85 / 99.70 / 98.85 | 26.42 / 26.45 / 26.83 | 69.5 / 72.2 / 70.8 |
| 0.30 | 98.90 | 26.76 | 68.0 |
| 0.25 | 99.30 / 101.05 / 98.60 | 27.18 / 26.69 / 27.34 | 70.3 / 71.6 / 71.6 |
| 0.20 | 98.60 | 27.04 | 71.7 |
| 0 (full 5-row windows) | 92.25 | 28.93 | 56.5 |
| 0 with `--spec 6` | 87.90 | 32.82 | 46.7 |
| 0.25 with `STRATA_SPEC_CUM=0.5` | 96.75 | 26.51 | 74.1 |

0.25 verifies ~2.7 rows per forward instead of 2.2: **+1.2% tok/s over three runs each (within noise)** for +1.9%
per forward. 0 is far too wide (4 rows, acceptance collapses to 56%). The automated form of this trade is
`STRATA_SPEC_COST=1` (§2d), which lands in the same place. Not deployed: no clear win, and it makes every forward
more expensive.

`STRATA_MTP_DRAFT_ALL` (the fork's one-graph drafting, the cut applied afterwards by the caller) was **not** ported:
at `--spec-min-p 0` our drafter already takes its one-graph path, and that arm is 3% *slower* than 0.5 - the extra
draft steps cost more than the host round trips they would save.

### 2f. The reference engine built here: 121 tok/s, and it is the combine fusions

The fork's tree was built on this box (0.1.43, `fc8f823`, ROCm 10.2, gfx1100) and run against *this* model and *this*
harness's documents (`ab/tg_fork.sh`, `ab/tg_sweep.py --port 8090`), 256 tokens, one fresh document per size. Upstream
`v0.1.43` (this repo's tag) was built too, and that same config was run against it:

| engine, same model / text / config | 32K TG | ms/fwd | 64K TG | ms/fwd |
|---|---|---|---|---|
| fork, their preset (42 switches) | **121.6** | 21.26 | **119.0** | 20.89 |
| fork, minus the `GR_*` group | 125.0 | 20.69 | 117.3 | 21.19 |
| fork, minus `RDNA_MMVQ_WAVE` / `RDNA_Q6K` | 119.3 | 21.03 | 117.1 | 21.03 |
| fork, minus `PF_*` / `PA_KLDS` / tile rows | 125.0 | 20.90 | 114.7 | 21.46 |
| fork, `--kv oct3 --kv-resident 65536` (our format) | 124.8 | 21.37 | 117.0 | 21.03 |
| **fork, no `COMBINE_CPU` / `WAIT_FUSE` / `ROUTE_PUBLISH`** | **92.6** | **28.50** | **93.1** | **28.35** |
| fork, no env at all | 94.5 | 27.64 | 90.9 | 27.88 |
| upstream `v0.1.43`, same args, 7 switches | 92.5 | 27.95 | 92.9 | 28.13 |
| ours (0.1.54 + `RDNA_Q6K`) | ~93 | ~27.5 | ~84 | ~27.5 |

So the whole 25% is **`COMBINE_CPU` + `WAIT_FUSE` + `ROUTE_PUBLISH`**: every other group and the KV format are inside
noise, upstream 0.1.43 without them is at 27.9 ms (the same as our 0.1.54), and the fork with them is at 21.0. That is
the fork's decode win, and it is the group ported in §2b.

**On our engine the same three switches measure 0.0 ms** (26.5 with, 26.4-26.8 without; 240 fewer graph nodes and no
change in wall time, with and without `STRATA_ADAPT_OVERLAP`). So the launch savings are not what their engine gains:
theirs must be the 27 MB per forward of CPU-row copies `copy_rows_from_mapped` makes (2.7 rows x 10 experts x 48
layers x 20 KB, over mapped host memory), which their unfused path pays and ours apparently does not - our deployed
decode hit rate is 94-96%, and the copies are already off the critical path here. The remaining 5.5 ms between our
fused window (26.5) and theirs (21.0) needs a per-kernel profile (rocprof-sys is on the box) rather than more
ablation: it is *not* launch count, the CPU share's throughput (4 workers cost 4%), or the graph's node count.

One piece of their `COMBINE_CPU` is still unported on our side: `STRATA_MMVF_AUX`, the shared expert's gate row riding
in the *multi-row* BF16 router GEMV with the gate applied inside the combine (`native_bf16.cu`'s multi kernel takes
the aux block there, ours only supports it in the rows kernel). It is one 2560-row matvec and a 2560-element sigmoid
per layer - unlikely to be the 5.5 ms, but it is the only code left in that group that they run and we do not.

### 2g. Where our own 26.5 ms goes (the engine's own stage table)

0.1.54 prints the window's GPU stage table into the engine log (ms per window, our deployed config, 130 windows):

| stage (GDN layers) | ms | stage (QSA layers) | ms |
|---|---|---|---|
| VRAM hits (expert GEMVs) | 2.63 | VRAM hits | 0.90 |
| **hyper-connection (hc0+hc1: norm/down/up, 6 launches)** | **3.30** | attention | 0.72 |
| shared+quant | 1.35 | q+q-idx | 0.78 |
| q8+qkv/q-idx gemv | 1.47 | router+plan+bell | 0.34 |
| out-proj | 1.02 | shared+quant | 0.45 |
| router+plan+bell | 1.06 | kv+idx append | 0.31 |
| head | 0.89 | scores+topk | 0.39 |
| waitCPU / copy+combine | 0.45 / 0.51 | waitCPU / copy+combine | 0.07 / 0.17 |
| PCIe grp | 0.38 | PCIe grp | 0.13 |
| **total, both layer kinds** | **22.95** | | |

The hyper-connection read is the largest single item after the experts: 3.3 ms for 317 MB of BF16 `hc_*` weights per
forward, i.e. **96 GB/s** - those six small launches are nowhere near bandwidth on this card, which is exactly what the
fork's `rdna_gr.cu` (2 launches, int8 weights, grid-barrier fused down/up) exists for. It is the next port; on the
reference engine the GR group made no difference, but their non-rdna GR is the same 0.1.43 path ours came from, and
the ablation cannot separate "GR is fast on both" from "GR is slow on both".

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

### 2h. The `GR_RDNA` read, and three config-level findings (2026-10-31)

**`STRATA_GR_RDNA=1`, ported as `src/kernels/cuda/rdna_gr.cu`** (the fork's BF16 half only; its int8/A8/grid-barrier
halves stayed behind). One split-K down launch (8 half-stream slices x 21 row groups of 16 rows) replaces the norm and
down launches, one up launch (160 blocks, four lanes per `w_up` row, the eight partials folded in the prologue)
replaces up+gate+mean. Same maths, another summation order. 32K, prod switches, lcz.me protocol:

| | tok/s | ms/window | acceptance |
|---|---|---|---|
| control | 98.1 | 27.47 | 73.2% |
| `GR_RDNA=1` | 102.1 | 26.39 | 73.5% |
| + PLE direct (below) | 103.9 | 25.94 | 73.5% |
| + `--mtp-window 4096` | 104.8 | 24.68 | 73.7% |

The stage table's six hyper-connection slots fall from 4.73 to ~3.0 ms/window (verified on the 130 with
`STRATA_VERIFY_PROFILE=1`, which costs ~6% by itself). 16-prompt harness (`win_tps2.sh`, 160K chars, 192 tokens):
**median 107.25 tok/s**, median 25.7 ms/window, acceptance 73.8% - against 95.6 / 29.5 / 67.3 for the deployed run
before this session, and 105.65 / 25.20 / 71.9 with PLE direct added (the acceptance column moves by +-2 points
between runs, which is most of the spread between those two medians).

**`--ple-io mmap` in the production config cost 0.4 ms/window.** The engine's default is `direct` with a 1M-row
cache; the config had pinned `mmap` (and `--ple-io` being explicit also turns off the auto-probe's fallback decision).
Its probe reads 70k rows/s; switching back to `direct` (+ `--ple-row-cache 1048576`) is -0.4 ms and no arithmetic
change.

**The unit's `Environment=STRATA_RDNA_Q6K=1` sat after `[Install]`**, where systemd ignores it: production had been
running without §2c's +3.5% since it was "deployed". It is now in `[Service]`, together with `STRATA_GR_RDNA=1`.

**The three decode fusions pay again once the window is cheaper.** §2b measured them as neutral (26.42 against
26.53 ms) on the pre-`GR_RDNA` window and kept them off. With the hyper-connection read down to two launches they are
worth -0.51 ms (25.95 -> 25.44, twice: 25.51 / 25.37, acceptance unchanged at 73.5%), i.e. **2.0%**: the launch and
wait overhead they remove is now a larger share of a smaller window. Deployed as `STRATA_COMBINE_CPU=1
STRATA_WAIT_FUSE=1 STRATA_ROUTE_PUBLISH=1`.

**Negative results, measured today** (all 32K, same protocol):

| arm | ms/window | acceptance | verdict |
|---|---|---|---|
| `--mtp-window 4096` / `8192` | 24.68 / 24.85 (vs 25.94) | 73.7% / 69.5% (vs 73.5%) | 1.1 ms cheaper but 0-4 acceptance points: 8192 is a net loss, 4096 a wash at 32K and worse at longer contexts. Not deployed |
| `--mtp-q4 all` | 26.46 (vs 26.39) | 74.0% | no gain: the draft is not weight-read bound |
| `STRATA_HC_Q8=1` (int8 hc weights, the existing S23 path) | 25.21 (vs 27.47) | 65.6% (vs 73.2%) | 2.3 ms cheaper (the read is bandwidth-bound: bf16 317 MB/window at 570 GB/s) but 7.6 acceptance points lost. 16-prompt median says the same: 101.75 tok/s / 24.53 ms / 68.0% against 105.65 / 25.20 / 71.9 - net loss. The MTP draft has its own hc weights (which stay bf16 either way), so this is the target's own distribution shifting, not a draft/target mismatch |
| `STRATA_VERIFY_PROFILE=1` | +0.5-2 ms | | the profiler's own cost (the fork's costs 2.4x: 121.6 -> 44.5 tok/s) |
| `STRATA_HC_REQ8=1` (the other int8 hc route: quantized at load, both reads) | 58.53 (vs 25.94) | 68.7% | 2.3x slower: the prompt path and the single-token reader dequantize the int8 rows into a scratch on every call, so the read is paid 2x plus a dequant per layer |
| `STRATA_NO_IQ512X1=1` (turns off the AVX512 one-token CPU expert kernel) | 26.09 (vs 25.94) | 73.5% | +0.15 ms only: the CPU share is hidden behind the GPU, so the fork's CPU IQ3_S byte-plane kernel has at most ~0.5% to give on this box - not worth the port (the engine already runs IQ3_S on AVX512) |
| `--spec-min-p 0.25` (the best of §2e's scan on the old window) | 26.58 (vs 25.94) | 66.4% | more drafts (238 against 219) and a dearer window now: the faster window did not make a lower floor pay. 0.5 stays |
| `--kv int8 --kv-resident 32768` (the fork's KV) | 26.92 (vs 25.94) | 72.4% | oct3 is both cheaper and just as accepted here: the draft is not KV-noise limited |
| `STRATA_Q8_PACKED=1` | 26.02 (vs 25.94) | 73.5% | neutral: the dense decode GEMV already runs in its own shape |

Status: `SPEC_CUM` (§2a), the three decode fusions (§2b), the K-quant GEMV (§2c) and the `GR_*` read (§2h) are ported
and measured; all of them are deployed now (the fusions only paid once `GR_RDNA` had shrunk the window: §2h),
`SPEC_CUM=0.4`, `STRATA_RDNA_Q6K=1`, `STRATA_GR_RDNA=1`, `COMBINE_CPU` / `WAIT_FUSE` / `ROUTE_PUBLISH`, PLE direct in
the config's args (§2h also fixed the unit line that had kept `RDNA_Q6K` off). §2d lists what did not
move the number. Still to do: the int8 `GR_RDNA` read (blocked on the quality loss in §2h), the wave-32 MMVQ layout
(`STRATA_RDNA_MMVQ_WAVE`), the CPU IQ3_S byte-plane kernel (mostly idle cores here: 4 pool workers cost only 4%), and
`PA_KLDS` / `PREFILL_MMQ_TILE_ROWS` (prefill).

### 2i. Kernel-level: rocprof is blind here, the benches are not (2026-10-31)

**rocprofv3 cannot see this engine's dispatches.** `rocprofv3 --attach <engine pid>` fails
(`rocattach_attach_tree failed ... error code 1`); launched as the parent it records only the runtime's own
`__amd_rocclr_copyBuffer` (204 calls) and `__amd_rocclr_fillBufferUnAligned` (96 calls) for a 30K prefill plus a
32-token decode, no compute kernel at all (the stats and the 41 KB kernel trace agree); `ROCPROF_KERNEL_TRACE` in the
engine's own environment behaves the same. The engine links the same `/opt/rocm/lib/libamdhip64.so.7` the profiler
interposes, so this is its dispatch path (graphs / its own queue), not the build. The engine's own stage stamps
(§2g) remain the per-stage measurement; the per-kernel numbers below come from the tree's benches, which run the
same kernels on the same weights.

`native_expert_bench <shard1.gguf> <layer> 10 <T> 0 0 100` - the routed-expert GEMV of one layer, 10 experts,
GPU-side, with the production kernel (mode 0):

| layer (gate/up, down) | T=1 | T=2 | T=3 | T=4 | T=6 | T=8 |
|---|---|---|---|---|---|---|
| **20** iq3_xxs / q2_0, all | 40 us | 44 | 47-53 | 56-59 | 91 | 100 |
| gu | 23-24 us (518-540 GB/s) | 26-28 (452-488) | 31 (407-410) | 32-34 (371-391) | 49-54 (233-258) | 56-58 (216-223) |
| down | 13-14 us (327-348 GB/s) | 15-16 (283-302) | 17-19 (248-272) | 19-21 (219-245) | 31-32 (144-147) | 36-37 (126-128) |
| **3 / 11 / 24** iq2_s / iq4_nl, all (T=3) | | | 47-54 | | | |
| gu (T=3) | | | 33-35 us (**295-315 GB/s**) | | | |
| down (T=3) | | | 17-18 us (501-540 GB/s) | | | |
| **47** iq4_xs / iq4_nl, all (T=3) | | | 45-50 | | | |
| gu (T=3) | | | 30 us (582-590 GB/s) | | | |

Mode 1 is the reference (not timed), mode 2 is the scalar path (2x slower: 112 us), mode 3 the older grouped layout
(50.6 us, a wash) - **mode 0, what production runs, is already the fastest of the three**.

At the deployed window (T ~2.5) that is ~48 us per layer; x 48 layers = **2.3 ms/window of pure expert GEMV**, against
the stage table's 3.15 + 1.04 ms "VRAM hits" - i.e. ~1.5 ms/window of per-layer bookkeeping (plan, cache, combine,
launches) rides on top of the GEMVs. The dense side (`q8_gemv_bench`, GDN qkv 2560x10240, 27.85 MB): 80.9 us at one
column (344 GB/s), 49.3 us at two (565 GB/s).

The draft's 3.0 ms/window splits (no profiler, so by construction and by the arms we have): the draft head is 106299
rows (213 MiB) read once per draft forward - ~0.35 ms at 600 GB/s; the draft's own MoE is one layer's GEMV (~50 us);
the rest is its attention over the whole 30K oct3 KV, which is exactly what `--mtp-window 4096` removes (-1.1 ms, at
0-4 acceptance points, §2h).

Where the headroom is, then: **iq2_s gate/up at ~300 GB/s** (the 36 GDN layers' experts) - a layout with more rows
per block / more waves would halve its 33 us and give back ~0.7 ms/window (2.7%); the expert bookkeeping (1.5 ms);
the draft's long-context attention (only by paying acceptance).

### 2j. The PCIe share's four empty nodes (2026-10-31 evening) - the biggest single win of the day

The stage table's `PCIe grp` (0.53 ms) and `waitB` (1.04 ms) were *pure overhead*: at `--pcie-frac 0` the pool never
plans a PCIe group, yet every layer still issued its staging wait (`wait_flag_ge(m_flagB_)`), `fetch_blobs`,
`rebase_ptrs` and the empty `grouped()` launch. Skipping the whole block (a `Verifier::set_pcie_share` from the
`--pcie-frac` the caller already knows; a balance model keeps it) gives, at 32K with the production switches:

| | ms_fwd (profiled) | stage total | acceptance |
|---|---|---|---|
| control | 27.28 | 23.47 | 73.5% |
| share skipped | **25.89** | **22.11** | 73.5% |

Unprofiled: **23.91 ms/window**; the 16-prompt harness (160K chars, 192 tokens, 16 prompts): median **112.55 tok/s**
(min 84.2, max 128.9), 23.53 ms/window, 73.0% acceptance - against 102.9 / 105.65 / 106.5 / 107.25 for the runs
before it. No arithmetic changes: no group was ever planned (the decode timing line reports `PCIe 0.00`).

**What it teaches:** a node in this window's graph costs **6-11 us** (four nodes = 1.4 ms of a 25.4 ms window), so
the node count matters more than the kernels' shapes. Two consequences measured today:

- `STRATA_GR_RDNA_RW` (1 / 2 / 4 rows per down block, ported from the fork) changes nothing: 0.66 ms either way.
  The hyper-connection read is **ALU-bound, not bandwidth-bound**: 6.55 MB per call at 2 instructions per
  (element x token) = ~18 us, and `v_dot2c_f32_bf16` - the only packed-bf16 dot - **the gfx1100 assembler rejects**
  (`instruction not supported on this GPU`), so bf16 stays 1 element an instruction (fp16/int8 would change the
  rounding; `STRATA_HC_Q8` costs 7.6 acceptance points, §2h).
- `STRATA_GR_FUSED=1` (down + up in one launch behind a grid barrier, the fork's shape) is **off on this card**: the
  driver reports 3 blocks x 48 multiprocessors = 144 co-resident blocks against the 168 the down grid needs, so its
  own guard declines rather than risking a deadlock. Making it fit means a 4-slice down grid (84 blocks) and a
  32-column up body (80 blocks) - not done.
- S26 `STRATA_LFUSE`'s gate+up pair (one launch for the shared expert's two Q8_0 GEMVs) is **neutral** here, and only
  when its `ar_on()` requirement is relaxed: shared+quant 1.20 -> 1.20 ms, total 22.11 -> 22.13, ms_fwd 25.89 ->
  25.92. A node is not a fixed 7 us: the four the PCIe share owned were a spin wait and a wide empty launch; a
  pair launch as wide as the two it replaces saves only its dispatch.

## 4. What we did not take, and why

- **The fork's tree as a base.** 0.1.43 against 0.1.54 is ~170 commits of upstream work (conversation snapshots,
  park VRAM slots, new prompt paths); starting from the fork would mean re-porting all of it. Everything above is
  small enough to move forward instead.
- **Defaults that change rounding silently.** Each port lands opt-in, like upstream's own new features.
- **Multi-GPU features** the fork carries (`STRATA_DP_FULL_RES`, `MTP_STAGE0`, ...): this box has one card.
