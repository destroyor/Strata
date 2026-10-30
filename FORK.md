# What this fork is

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), branched from **v0.2.8** (engine 0.1.54), carrying
what this box uses and upstream does not have (or leaves off on gfx1100):

| branch | what | state |
|---|---|---|
| `oct3` | `--kv oct3`: K/V as 3-bit Lloyd-Max codes with an exact fp16 window (4 sinks + 1024 newest cells per layer), 448 B/cell | measured on one RX 7900 XTX: quality, prefill, decode, long-context needles; deployed |
| `fork-accel` | on top of `oct3`: `STRATA_SPEC_CUM` - a cumulative draft-probability floor for the MTP window, ported from nicejunjie/Strata-rdna3 | measured: +3.3% decode tok/s at 0.4 (two runs against two base runs) |
| `gfx1100-mmvf-rows1` | gfx1100 decode tuning: one row per block for the q8_0 GEMV, `STRATA_MMVF_RPB` for the bf16 multi-row GEMV | measured on engine 0.1.49.2, ported and compiled on v0.2.8, **not re-measured there** |

`main` is untouched upstream. Nothing here changes a default of the path it does not name: without `--kv oct3` the
engine runs exactly what upstream runs, and `oct3` is opt-in.

Also measured, opt-in and not a branch: `perf/presets/gfx1100-prefill.env`, upstream's gfx1151/gfx12 switches on
gfx1100, +13% prefill at 32K with a prompt-path KL of 0.0138 (`docs/FORK-RDNA3-MAPPING.md` §1).

## Where the numbers and the instructions are

| file | contents |
|---|---|
| [docs/OCT3.md](docs/OCT3.md) | what the format is, the window, the knobs, the measured quality, what it does not do |
| [docs/PERF.md](docs/PERF.md) | every measurement quoted above: method, configs, tables, noise |
| [docs/DEPLOY.md](docs/DEPLOY.md) | the build, the service, the config, the rollback, on this one machine |
| [docs/FORK-RDNA3-MAPPING.md](docs/FORK-RDNA3-MAPPING.md) | what `nicejunjie/Strata-rdna3` changes, which of it upstream already has (and switches off on gfx1100), what is left to port |
| `diffs/` | the patches themselves |
| `perf/` | raw runs (`perf/raw/`), the harness that produced them (`perf/harness/`), the medians (`perf/tps2-medians.tsv`), the quality tables (`perf/kl-tables.txt`) |

## diffs/

| patch | base | applies to |
|---|---|---|
| `oct3-v0.2.8.patch` | v0.2.8 | the port in this repo (both `oct3` commits, 22 files, 6 of them new) |
| `oct3-against-v0.2.4.2.patch` | v0.2.4.2 | the original tree the format was written in, before the port |
| `gfx1100-mmvf-rows1-v0.2.8.patch` | v0.2.8 | the decode-tuning branch |

## Building

Same as upstream; see [docs/DEPLOY.md](docs/DEPLOY.md) for the exact flags used on the RX 7900 XTX (HIP, gfx1100).
The port itself was built on the box: 368 targets, `strata`, `kv_octave_parity`, `kv_exact_window_parity`,
`kv_stream_parity` all link, and the two oct3 parity programs pass there.
