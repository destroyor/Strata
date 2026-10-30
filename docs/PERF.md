# Measurements

Every number here was measured on one machine, on the same day as the build it describes. Nothing is an estimate.

## The machine

| | |
|---|---|
| GPU | Radeon RX 7900 XTX, gfx1100 (RDNA3), 24 GB |
| CPU / RAM | Ryzen 9 9950X (16 cores), 91.7 GiB, no swap |
| OS / ROCm | EndeavourOS / Arch, kernel 7.2.10, ROCm nightly `gfx110x` 10.2.0a20261016, hipBLASLt 1.0.0.5 |
| Model | Qwen3.8-Flash-Next GSQ-RCO abliterated IQ3_S (125B MoE / 6B active + MTP draft head) |
| Engine | 0.1.49.2 (v0.2.4.2) for the runs below; the v0.2.8 port is compiled and booted on the same box |
| Decode | `--spec 5 --spec-min-p 0.5`, `--pool-workers 15` (one per physical core; oversubscription costs 25% of decode) |

## How it was run

Two harnesses, both in `perf/harness/` and both driven through the OpenAI endpoint of a real server:

- `win_perf.sh` → `ctx_curve.py`: one cold prefix per size (32K, 96K), 192 generated tokens, 4 reps; rep 0 warms the
  expert cache and is dropped. Yields `prompt_tps`, ms per forward, tok/s, draft acceptance. GPU clock and power are
  sampled throughout so a throttled run is visible. Output: `perf/raw/perf-*.jsonl`.
- `win_tps2.sh` → `win_tps2.py`: 16 short prompts, 192 tokens each, a warm 160K-char rolling prefix; reports the
  median over prompts. Output: `perf/raw/tps2-*.txt`, collected in `perf/tps2-medians.tsv`.
- `win_needle.sh` → `tools/needle_bench.py`: 96K/128K needles at depth 50. Output: `perf/raw/needles-*.json`.

Every arm stops the production unit, starts its own engine, measures, and stops it. The production unit was restarted
and checked (`/health` 200) after each batch.

## Prefill

`perf/raw/perf-pf*.jsonl`, oct3, 4 reps per size, median:

| arm | `--prefill` | `--kv-resident` | 32K prefill | 96K prefill |
|---|---|---|---|---|
| `perf-pf8-r96` | `auto:8192` | 65536 | 1456 tok/s | 1376 tok/s |
| `perf-pf32-r96` | `auto:32768` | 65536 | 1880 tok/s | 1794 tok/s |
| `perf-pf32-r384` | `auto:32768` | 65536 | 1881 tok/s | 1801 tok/s |

A 32K prompt chunk is **+29.1% / +30.4%** prefill over 8K at the same KV residency; the staging ring (96 against 384)
makes no difference. This is the change that went to production first.

Format A/B, same config except `--kv` (`perf-oct3b` against `perf-k8v4b`, `--prefill auto:8192`,
`--kv-resident 32768`): 1458 / 1378 (oct3) against 1461 / 1373 (k8v4) - the format is free at prefill.

## Decode

ms per forward, same arms as above (4 reps, median):

| arm | 32K | 96K |
|---|---|---|
| oct3, 8K chunk, resident 32768 | 25.64 | 26.17 |
| k8v4, 8K chunk, resident 32768 | 24.91 | 25.55 |
| oct3, 32K chunk, resident 65536 (production) | 25.00 | 25.51 |
| oct3, 32K chunk, resident 65536 + the fork's upstream knobs | 26.08 | 25.23 |

The 16-prompt harness, median tok/s over 16 prompts (`perf/tps2-medians.tsv` has all of them with config and env):

| chars of prefix | k8v4 | oct3 | notes |
|---|---|---|---|
| 80,000 | 93.20 | **96.55** | oct3 faster here |
| 160,000 | 90.80 / 92.50 / 91.85 | 90.75 / 90.45 / 90.40 | three runs each, interleaved |
| 213,000 | 88.35 / 88.90 | 87.10 / 91.00 | resident 65536 for the second oct3 arm |
| 320,000 | 88.00 | 88.40 / 90.20 | the second oct3 arm at `--kv-resident 65536` |

The spread between repeats of one arm (88.35 against 88.90; 89.55 against 90.65) is larger than the gap between the
formats: **oct3 and k8v4 decode at the same speed within noise**, and the draft acceptance explains the rest (oct3
60.0-62.1%, k8v4 63.1%).

The 16-prompt harness with the fork's upstream knobs (`STRATA_PREFILL_RING=192 STRATA_PREFILL_CPU_SHARE=auto
STRATA_PREFILL_CPU_SHARE_MAX=6144 STRATA_QFUSE=1 STRATA_PA_FAST=1 STRATA_HC_UPMIX=1 STRATA_HEAD_MIX_MULTI=1
STRATA_VERIFY_QDEDUP=1 STRATA_KV_HOST_DMA=1 STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_PAD=1
STRATA_PF_HCDOWN=1`) measured 93.85 tok/s median against 90.4-90.8 for oct3 alone on the same harness, and
2104 / 2000 tok/s prefill at 32K / 96K against 1880 / 1794. **This arm is not deployed**: it changes prefill
rounding (PF_GEMM, PF_FUSED, PA_FAST) and its quality has not been checked with `tools/quant_quality.py`'s KL
yardstick.

## Speculative window floor (`STRATA_SPEC_CUM`, fork port)

16-prompt harness, production config (`--spec 5 --spec-min-p 0.5`), oct3, two runs per arm at the extremes,
`perf/raw/tps2-facc-*.txt`:

| arm | tok/s (median of 16 prompts) | ms per forward | acceptance |
|---|---|---|---|
| off | 90.00 / 90.65 | 30.32 / 30.42 | 60.0% / 60.8% |
| 0.25 | 92.15 | 30.64 | 61.8% |
| 0.4 | 93.05 / 93.55 | 30.49 / 30.36 | 66.3% / 66.3% |
| 0.6 | 90.00 | 28.01 | 72.8% |

+3.3% median throughput at 0.4, the cost per forward unchanged, acceptance +6 points. 0.6 ends the windows early
enough that the gain is gone. The floor only chooses which drafts are verified; the sample is unchanged, so no KL run
is needed for it (unlike the prefill switches above).

## The RDNA3 fork's decode fusions (`ROUTE_PUBLISH` / `COMBINE_CPU` / `WAIT_FUSE`), opt-in and off

Three launches per MoE layer folded away, all bitwise (`docs/FORK-RDNA3-MAPPING.md` §2b). 16-prompt harness,
production config and its 23 switches, engine 0.1.54, binary `40a61eaa`, `perf/raw/tps2-b-c*.txt`:

| arm | tok/s | ms/fwd | acceptance |
|---|---|---|---|
| control | 96.05 / 98.20 | 26.97 / 27.04 | 73.0% / 74.2% |
| `ROUTE_PUBLISH=1` | 97.55 | 27.20 | 72.9% |
| `COMBINE_CPU=1` | 97.85 | 27.10 | 72.9% |
| `COMBINE_CPU=1 FUSE_WAIT=1` | 97.50 / 96.50 / 97.30 | 27.10 / 26.67 / 26.90 | 72.5% / 69.8% / 72.0% |

The harness repeats within ±1.5% (the two control runs differ by 2.2%), and a paired per-prompt comparison puts the
arms between -1.5% and +1.1% of the control. The fusions are real (ms per forward is 0.2-2.2% lower for
`COMBINE_CPU`) but too small for this host to show as throughput, so they are not deployed. Greedy answers are
character-identical on and off.

## On the v0.2.8 port (engine 0.1.54)

The same config on both engines (`--kv-resident 32768`, `--prefill auto:8192`), only `--kv` and the binary differ,
4 reps, `perf/raw/perf-k8v4b.jsonl`, `perf-oct3b.jsonl`, `perf-u54-k8v4.jsonl`, `perf-u54-oct3.jsonl`:

| engine | KV | 32K prefill | 96K prefill | 32K ms/fwd | 96K ms/fwd |
|---|---|---|---|---|---|
| 0.1.49.2 | k8v4 | 1458 tok/s | 1373 tok/s | 24.91 | 25.55 |
| 0.1.49.2 | oct3 | 1458 | 1378 | 25.64 | 26.17 |
| 0.1.54 (this repo) | k8v4 | 1468 | 1377 | 26.14 | 25.60 |
| 0.1.54 (this repo) | oct3 | 1456 | 1378 | 25.82 | 25.14 |

**The port costs nothing measurable**: prefill is the same across the two engines and the two formats, and ms per
forward is within 1 ms in all four cells. The production config (`--kv-resident 65536 --prefill auto:32768`) measured
28.22 / 26.68 ms per forward on 0.1.54 in `perf-fa54-base.jsonl`, higher than any cell here, but that arm's four reps
span 25.2-31.4 ms - one run of a noisier configuration, not used to claim a difference either way.

The switch set of `docs/FORK-RDNA3-MAPPING.md` §1 adds +13% / +12% prefill on top of the 0.1.54 row
(`perf-fa54-knobs.jsonl`: 2108 / 2008 tok/s, 25.33 / 26.11 ms per forward). It is available as an opt-in environment
file, `perf/presets/gfx1100-prefill.env`, with the quality numbers that come with it.

## Deployed on the box (2026-10-30)

Engine 0.1.54 from this repo (`oct3` + `STRATA_SPEC_CUM` + the rows1 commit), the unit's 22 switches, production
config (`--kv oct3 --kv-resident 65536 --prefill auto:32768 --spec 5 --spec-min-p 0.5`), `STRATA_SPEC_CUM=0.4`.

| stage | prefill 32K | prefill 96K | 16-prompt decode |
|---|---|---|---|
| 0.1.49.2 production (the old engine) | 1880 | 1794 | 90.00 / 90.65 |
| 0.1.54, no switches | 1864 | 1787 | 90.00 |
| + the ten arithmetic switches and the staging ring (13) | 2108 | 2008 | 90.4-90.8 |
| + `STRATA_SPEC_CUM=0.4` | 2108 | 2008 | 93.05 / 93.55 |
| + `HIP_WMMA` | 2090 | 1928 | - |
| + GDN / `PF_F16` / `PREFILL_AUTO_MAX` / `STAGER_RING` (the full set) | 2243 | 1979 | - |
| + `SELECT_WMMA` (QSA block scorer on matrix cores) | **2280** | **2304** | - |
| **deployed: full set + `SPEC_CUM=0.4` + `SELECT_WMMA`** | **2280** | **2304** | **95.60** (0 stalls, `perf/raw/tps2-prod-deployed.txt`) |

`SELECT_WMMA` is the long-context half of the prefill win: +1.6% at 32K, **+16.4% at 96K** (the block scorer's work
grows with the context), and its selected ids are identical to the warp scorer's on this model - 29,995 positions over
five 6,000-token segments, top-1 100.00%, KL 0.00000 (`/tmp/qqk6-base.tsv` against `qqk6-sel.tsv`, full switch set on
both sides).

The deployed service, measured through its own API on cold prompts (no cache reuse): 1,971 / 2,010 / 2,012 tok/s at
35,946 tokens (the first request of a fresh engine 1,529, library warm-up), and 2,172 / 2,241 tok/s at 109,281 tokens.
Needles on the same service: 96K and 128K at depth 50 both found, 2 of 2 (`perf/raw/needles-deployed.json`).

Where the prefill comes from, switch group by switch group: the prompt fusion/GEMM/CPU-share group is +13%, the GDN
chunked recurrence and the staging-ring/auto-chunk group another +6%, the matrix-core block scorer +16% at 96K (and
+1.6% at 32K), and `HIP_WMMA` (the prompt attention on matrix cores) nothing measurable on this model - the log does
confirm it engages (`strata: STRATA_HIP_WMMA: the prompt attention on matrix cores on (gfx1100)`).

**One crash, understood**: a run with `--stats` stalled at the first prompt chunk's layer 0, and the engine's own
watchdog aborted it 60 s later (`Signal 6 (ABRT) si_code: SI_TKILL`, thread `strata-watchdog`; the serve log says
`no progress for 60 s ... stopping the engine so the server starts it again (issue #29)`). It left a 31.6 GB core
because that test ran as root (the unit sets `LimitCORE=0`). Not reproduced in four other runs of the same switches
without `--stats`, including two 38.6K-token prompts in one session and the 16-prompt harness above.

## Where the last 1.5x of prefill is: the prompt-attention kernel

The forum's 3.5K was measured with `--kv int8`. On this engine, with the same 23 switches, the KV format alone decided
whether the batched (matrix-core) prompt attention ran at all - `qsa_prompt_attn_batch` on AMD took only int8 pools
(`k_q`/`v_q`/`k_scale`/`v_scale`, no `k_q4`), everything else fell back to the decode kernel's chunk path, one window
of queries at a time. 4 reps, production config, `perf/raw/perf-fmt-*.jsonl`:

| KV | 32K prefill | 96K prefill | 16-prompt decode | cell size |
|---|---|---|---|---|
| oct3, before `STRATA_PA_OCT` | 2280 | 2304 | 95.60 tok/s | 448 B |
| k8v4 | 2307 | 2313 | - | 816 B |
| **int8** | **3506** | **3559** | 90.75 tok/s | 1056 B |
| **oct3 + `STRATA_PA_OCT=1`** | **3443** | **3513** | - | 448 B |

`STRATA_PA_OCT=1` (docs/OCT3.md) is the oct3 arm of that kernel: the 3-bit codes are expanded in registers (bit-plane
nibble spread + a `v_perm_b32` codebook lookup) and the per-32-dim scales are applied after the MMA, two per wave. The
re-run below is one session, 4 reps (`perf/raw/perf-int8ref.jsonl`, `perf/raw/perf-paoct2.jsonl`):

| KV | 32K prefill | 96K prefill |
|---|---|---|
| oct3, no `STRATA_PA_OCT` (the 0.1.54 deployment) | 2298 | 2305 |
| `--kv int8` | 3516 | 3557 |
| **oct3 + `STRATA_PA_OCT=1`** | **3443** | **3513** |

So the 3-bit format now prefills within 1-2% of int8 and at the forum's 3.5K, keeping 448 B/cell (2.4x fewer bytes
than int8) and its ~5% decode advantage. Kernel level (synthetic, 2,048 queries over a 32K context, `qsa_prompt_attn_parity`):
8.59 ms per chunk against int8's 8.05, 4.2x the FP32 chunk path it replaces; its output is the FP32 kernel's to
2.0e-9 (7.9e-7 of scale) and the FP64 reference's to 1.1e-9.

The deployed service with the switch in its unit (the same production config, `perf/raw/perf-deployed-paoct.jsonl`):
32,028 / 31,796 / 32,010-token prompts at **3,457 / 3,455 / 3,444 tok/s**, 95,937 / 96,155 / 95,967-token prompts at
**3,504 / 3,507 / 3,505 tok/s**; 96K needles 3 of 3 at depths 10/50/90. Decode is untouched - the prompt attention
does not run there.

Decode, the same 16-prompt harness on the deployed binary (`perf/raw/tps2-decout-*.txt`), three runs with the switch
and one without: **ms per forward 29.96 / 30.02 / 30.44 against 30.22** - the step itself is unchanged, as the decode
kernels are. Throughput reads 92.0 / 90.3 / 94.45 tok/s against 96.25, all of it draft acceptance (66.3 / 66.6 /
68.1% against 68.9%): `STRATA_PA_FAST=1` moves the prompt attention by 3.5e-4, which the drafter's acceptance sees at
that scale. `STRATA_PA_FAST=0` (fp32-level, 7e-7) brings it back and costs the prefill path ~5% (3,280 / 3,320).

## Quality

`perf/kl-tables.txt` (produced by `tools/quant_quality.py compare`), repeated in `docs/OCT3.md`. Reference is the same
engine at fp16 K/V.

## Long context

96K and 128K needles at depth 50, `perf/raw/needles-win-*.json`:

| run | 96K | 128K |
|---|---|---|
| `needles-win-pf32c1/2/3` (oct3, window on) | found | found |
| `needles-win-pf32n1/2/3` (oct3, window off) | found, **missed**, found | - , found, found |
| `needles-win-win0/win1` (oct3, window off / on) | found | found |

One miss in eleven long-context samples, in a window-off arm. k8v4 finds 32K and 128K at depths 10/50/90
(`needles-k8v4.json`).

## Parity tests

Built with the engine and run on the same card:

- `kv_octave_parity` - device bytes against the host reference, bit for bit; error on rotated vectors against a 4-bit
  grid (oct3 MSE/coord 0.0508, q4_0 0.0114, 4.47x at 78% of the bytes); the engine-shaped storage round trip.
- `kv_exact_window_parity` - eviction order, hits, exact values; the window's cost at the real geometry.
- `kv_stream_parity` - every KV format through the streamed and resident pools, oct3 included.

## What is not measured

- The v0.2.8 port was booted, answered prompts (76 tok/s cold on a short prompt) and sized the window (24.1 MiB,
  12 layers, 710 MiB VRAM free), but the tables above are engine 0.1.49.2. The port is upstream's code plus this
  format; a re-run of `win_perf.sh` on 0.1.54 is the next step, not a claim here.
- The `gfx1100-mmvf-rows1` branch: measured on 0.1.49.2 (23.355 → 23.15 ms per forward, identical `sha_set`), compiled
  on v0.2.8, not re-measured there.
- Multi-GPU, other cards, other models.
