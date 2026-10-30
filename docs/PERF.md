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
