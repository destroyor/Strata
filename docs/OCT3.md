# `--kv oct3`: 3-bit Lloyd-Max K/V with an exact fp16 window

Opt-in. Without `--kv oct3` the engine runs the code it ran before; the format touches no other path.

## The problem it answers

A QSA layer's K/V is streamed from pinned RAM (`--kv-resident`) and read back through a sparse top-k selection. At
3 bits per value the codes are small (448 B per cell: 55% of k8v4's 816, 78% of q4_0's 576), but the selection is not
uniform: 35-80% of the attention mass lands on a handful of recent cells, and their quantization error does not
average out over the sequence. The error shows up as a worse next-token distribution, not as noise.

The fix is the one vLLM's Octave KV reports: keep the cells that carry the mass in fp16 beside the codes.

## The two pieces

**`kv_octave.hpp` / `kv_octave.cu`** - K and V stored as 3-bit Lloyd-Max codes over a 256-value block, after the same
Walsh-Hadamard rotation Q4_0 uses (`kv_q4.hpp`). The levels are the Lloyd-Max ones for a unit Gaussian, scaled by a
per-block fp16 `d`; 14 bytes per block (2 scale + 12 codes), 112 bytes per head per K or V. Appends and gathers are
shaped like `kv_q4.cu`'s, so the streaming map, the host copy, the staging pool and the snapshot code treat an oct3
layer as "two byte runs at 14-byte blocks" (`KvFormat::kKvOct3`).

**`kv_exact_window.hpp` / `kv_exact_window.cu`** - a per-layer ring of `sinks + tail` rows in fp16, with a 4-byte tag
per row holding the position a row currently answers for. The append path writes the same rotated K/V into it; the
decode attention reads a selected cell from the window when its tag matches, from the codes otherwise. A row that a
newer position has reused does not match, so a stale row is a miss, never a wrong value.

The window is **not part of a snapshot**: it is a cache. A restore or a copy into another state invalidates every
tag (`qsa_window_invalidate`), and the rows refill as the sequence appends.

## Knobs

| knob | default | meaning |
|---|---|---|
| `--kv oct3` | off | the format |
| `STRATA_KV_OCT=1` | - | the same format without the `--kv` name |
| `STRATA_KV_WINDOW=0` | on | turns the window off (pure 3-bit, for A/B) |
| `STRATA_KV_WINDOW_SINKS=N` | 4 | leading cells kept in fp16 |
| `STRATA_KV_WINDOW_TAIL=N` | 1024 | newest cells kept in fp16 |

The startup log names the window the session actually sized:

```
strata generate: --kv oct3: 3-bit Lloyd-Max K/V (448 B/cell, gather path; an fp16 window of 4 sinks + 1024 tail cells per layer)
strata generate: KV window: the first 4 and the newest 1024 cells of every QSA layer in fp16 (24.1 MiB over 12 layers)
```

## Quality

Reference: the same engine and model at fp16 K/V. `tools/quant_quality.py` (the repo's own yardstick), 5 x 6000-token
segments, 29,995 positions, next-token distribution per position:

| K/V | top-1 agreement | mean KL to fp16 |
|---|---|---|
| oct3, no window | 91.63% | 0.05261 |
| oct3, 4 + 64 | 93.62% | 0.03342 |
| oct3, 4 + 256 | 94.24% | 0.02559 |
| **oct3, 4 + 1024 (default)** | **95.27%** | **0.01851** |
| oct3, window on K only | 92.42% | 0.04207 |

Each doubling of the tail is cheaper in quality than the last, and 1024 cells cost 12 x 2 MiB = 24.1 MiB of VRAM -
3.6% of what the same cells would cost in an fp16 cache.

On a second, shorter run (4,995 positions) the formats are compared directly:

| K/V | top-1 | KL |
|---|---|---|
| k8v4 (INT8 K + rotated Q4_0 V) | 93.31% | 0.03961 |
| oct3, no window | 89.75% | 0.08623 |
| oct3, 4 + 32 | 92.19% | 0.05506 |
| oct3, 4 + 128 | 93.59% | 0.04454 |
| oct3, 4 + 256 | 94.03% | 0.03704 |

So at 256 tail cells oct3 is already below k8v4's KL at half its bytes; the 1024 default buys the rest.

Long context (`tools/needle_bench.py`, this model, single card): 96K and 128K at depth 50 both found in the runs
recorded under `perf/raw/needles-win-*.json`. One 96K run missed (`needles-win-pf32n1.json`, a single sample); k8v4
on the same box finds 32K and 128K at depths 10/50/90 (`needles-k8v4.json`).

## Cost

Measured on one RX 7900 XTX, Qwen3.8-Flash-Next GSQ-RCO IQ3_S. The two formats were run back to back on the same
config except `--kv` (`perf/raw/perf-oct3b.jsonl`, `perf/raw/perf-k8v4b.jsonl`, `--kv-resident 32768`,
`--prefill auto:8192`, 4 reps per size):

| | oct3 | k8v4 |
|---|---|---|
| prefill 32K | 1458 tok/s | 1461 tok/s |
| prefill 96K | 1378 tok/s | 1373 tok/s |
| decode, ms per forward, 32K / 96K | 25.64 / 26.17 | 24.91 / 25.55 |
| draft acceptance (16-prompt harness) | 60.0-62.1% | 63.1% |

Prefill is the same. Decode per forward is 2-3% higher in that pair and lower in others (`docs/PERF.md` has every
run); the throughput is 1-3% lower because a 3-bit K loses a point or two of draft acceptance against an 8-bit one.
The production config (oct3, `--kv-resident 65536`, `--prefill auto:32768`) measures 1880 tok/s prefill at 32K,
1794 at 96K, 25.0 / 25.5 ms per forward.

## What it does not do

- **Gather path only.** There is no fused decode-attention mode for oct3 (`qsa_decode_attn.cu` falls back to the
  chunk kernel with `KV_MODE == 4`), no prompt-attention mode (`qsa_prompt_attn_batch` returns false), and no
  row-batched or slot-batched attention (those exit with a message if asked for oct3). `--kv oct3` with
  `STRATA_BATCH_ATTN_ROWS` or a multi-slot server is refused, not silently misread.
- **The MTP drafter stays on plain INT8** (`mtp.cpp`), for the same reason it does under k8v4.
- **Not a snapshot format.** A slot image never contains the window; restoring invalidates it.
- **Experimental.** The format is not a default and upstream has not accepted it.

## Files

New: `include/strata/kernels/kv_octave.hpp`, `src/kernels/cuda/kv_octave.cu`, `include/strata/kernels/kv_exact_window.hpp`,
`src/kernels/cuda/kv_exact_window.cu`, `src/kernels/kv_octave_parity.cpp`, `src/kernels/kv_exact_window_parity.cpp`.

Touched: `CMakeLists.txt`, `include/strata/core/layer.hpp`, `include/strata/kernels/kv_stream.hpp`,
`include/strata/kernels/qsa_decode_attn.hpp`, `include/strata/prefill/prefill.hpp`, `src/core/layer.cpp`,
`src/core/verify.cpp`, `src/core/conversation_snapshot.cpp`, `src/core/mtp.cpp`, `src/kernels/cuda/kv_stream.cu`,
`src/kernels/cuda/qsa_decode_attn.cu`, `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/kv_stream_parity.cpp`,
`src/prefill/prefill.cpp`, `src/program/generate.cpp`, `tools/quant_quality.py` (`--kv-format`, so the yardstick can
measure a KV format).

The patch is `diffs/oct3-v0.2.8.patch`.
