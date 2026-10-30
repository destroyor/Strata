# Deploying this fork on the test box (RX 7900 XTX, gfx1100)

Everything below was run on `192.168.100.130` (`des-eliteminiseries`), the machine all of `docs/PERF.md` was measured
on. Paths are this box's; change them for another one.

## 0. Layout

```
/home/des/test/strata/                       upstream checkout (v0.2.4.2 / engine 0.1.49.2), the production engine
/home/des/test/strata/engine/strata          the binary the service runs
/home/des/test/strata/strata-sc117-iq3_s.json  the service config (args + paths)
/home/des/test/strata/.venv/                 the Python server environment (`serve/`)
/home/des/test/strata/ab/                    the benchmark harness and its raw output
/home/des/test/strata/third_party/llama.cpp/ ggml, shared with every build
```

The fork in this repo builds beside it, never on top of it: `build-hip-oct3/strata` is a different binary.

## 1. Build

```bash
cd /path/to/strata-fork                 # this repo, branch oct3
git checkout oct3

PATH=/opt/rocm/bin:$PATH HIP_PATH=/opt/rocm \
cmake -G Ninja -S . -B build-hip-oct3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF \
  -DSTRATA_GGML_DIR=/home/des/test/strata/third_party/llama.cpp
cmake --build build-hip-oct3 -j20      # 368 targets, 4-6 min on 16 cores
```

Notes:

- `-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++` is required; without it CMake fails with
  `Failed to find HIP root directory`.
- `STRATA_GGML_DIR` points at an existing llama.cpp checkout so the build does not fetch ggml again. Its commit is
  pinned by `third_party/ggml/VERSION.txt`-style checks in `CMakeLists.txt`; the checkout used here is the one the
  production engine was built from.
- `STRATA_BUILD_DSV41` stays off (DeepSeek-V4.1 path, not used here).
- The vision front end is a separate project (`tools/vision`) and is not needed to serve text; the production unit
  uses the 0.1.49.2 `strata-vision` binary beside the engine.

The build produces `strata`, `kv_octave_parity`, `kv_exact_window_parity`, `kv_stream_parity`, `q8_gemv_bench` and the
rest of upstream's tools.

## 2. Run the parity tests (they need the card, not the model)

```bash
cd /path/to/strata-fork
./build-hip-oct3/kv_octave_parity          # device bytes == host reference; oct3 against a 4-bit grid
./build-hip-oct3/kv_exact_window_parity    # eviction order, hits, exact values, window cost
./build-hip-oct3/kv_stream_parity          # every format, streamed and resident pools
```

They allocate a few hundred MiB. With the production engine loaded there is not enough VRAM free; run them with the
service stopped.

## 3. The service

User unit `strata-sc117` (owner `des`), OpenAI-compatible server on **8089**:

```ini
[Service]
WorkingDirectory=/home/des/test/strata
ExecStart=/home/des/test/strata/.venv/bin/python -m serve.server \
          --engine strata --config strata-sc117-iq3_s.json --port 8089
Restart=on-failure
RestartSec=15
Environment=LD_LIBRARY_PATH=/opt/rocm/lib
Environment=STRATA_ADAPT_OVERLAP=1
Environment=STRATA_PREFILL_MMQ=1
Environment=STRATA_HIPBLASLT_TUNING=/home/des/test/strata/tools/hip/gfx1100-hipblaslt-100500-20261016.txt
Environment=STRATA_PREFILL_RING=96
Environment=STRATA_IO_THREADS=32
```

The engine arguments (`strata-sc117-iq3_s.json` → `args`), the ones that matter here:

```
--pack /home/des/test/Strata-data/packs/sc117-iq3_s
--native .../Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf
--ple-gguf .../Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf
--expert-profile data/expert-profile.bin --expert-cache 7700
--prefill auto:32768          # the chunk that is worth +29% prefill over 8K
--spec 5 --spec-min-p 0.5 --mtp .../strata/rt
--max-context 262144
--kv oct3                     # this fork
--kv-resident 65536           # 32768 regresses streaming decode between 48K and 64K cells
--vram-reserve-mib 1024 --adapt-every 4 --pcie-frac 0
--pool-workers 15             # one per physical core; oversubscription costs ~25% of decode
--vision
--prompt-cache 16 --prompt-cache-every 16384
--conversation-cache-slots 8 --conversation-cache-mib 16384 --conversation-cache-min-tokens 32768
--ple-io mmap
```

Memory at load (engine 0.1.49.2): experts 46.84 GiB pinned in RAM, expert cache 8,883 slots / 16.84 GiB VRAM,
KV streaming 65,536 of 262,144 cells per QSA layer plus 2.39 GiB of pinned K/V, the oct3 window 24.1 MiB over 12
layers. `710 MiB of VRAM free with everything loaded` on the v0.2.8 build; the verify graph's capture then takes
~550 MiB of that. **Plan capacity with the post-capture number, not the load-time one.**

Optional and measured, not enabled by default: [`perf/presets/gfx1100-prefill.env`](../perf/presets/gfx1100-prefill.env)
- upstream's gfx1151/gfx12 switches, +13% prefill at 32K on this box, at a prompt-path KL of 0.0138 against the
unswitched run (`docs/FORK-RDNA3-MAPPING.md` §1 has the numbers and the one unreproduced hang). Add its lines to the
unit's `Environment=` only if that trade is wanted.

## 4. Start, stop, verify

```bash
su - des -c "XDG_RUNTIME_DIR=/run/user/1000 systemctl --user start|stop|restart strata-sc117"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8089/health    # 200 after 30-60 s
```

Lines that say the fork is in effect:

```
strata generate: --kv oct3: 3-bit Lloyd-Max K/V (448 B/cell, gather path; an fp16 window of 4 sinks + 1024 tail cells per layer)
strata generate: KV window: the first 4 and the newest 1024 cells of every QSA layer in fp16 (24.1 MiB over 12 layers)
strata generate: session is up (engine 0.1.54)
```

Two upstream lines that say how the memory is being used, and are the first thing to read when a change looks wrong:

```
strata serve: decode expert cache hit rate: 86.2% (125336 hits / 145440 lookups)
strata serve: KV streaming: 95.50% of 1417296 block reads hit VRAM, 256.8 MiB read from RAM
```

## 5. Measuring on this box (and not leaving the service down)

`perf/harness/win_perf.sh`, `win_tps2.sh`, `win_needle.sh` each **stop `strata-sc117`, start their own engine from
`ab/launch.sh` + `$AB_CFG`, and kill it at the end without restarting the unit.** After any batch:

```bash
su - des -c "XDG_RUNTIME_DIR=/run/user/1000 systemctl --user start strata-sc117"
# and wait for /health 200 before calling it done
```

The environment for an arm goes through `/tmp/ab_env` (one `K=V` per line), which `launch.sh` sources; the config is
`$AB_CFG` (absolute path, e.g. `/tmp/cfg-oct3.json`). The production config is
`/home/des/test/strata/strata-sc117-iq3_s.json`; copies differing only in `--kv` / `--kv-resident` / `--prefill` are
what the A/Bs used.

`ulimit -c 0` in `launch.sh`: a core dump of this engine is ~25 GB and the box has no swap.

## 6. Rollback

| what | how |
|---|---|
| back to the upstream engine | the service unit's `engine/strata` is untouched by this fork; stop the unit, `install build-hip-oct3/strata engine/strata` only if you meant to replace it |
| back to k8v4 | edit `--kv oct3` → `--kv k8v4` in `strata-sc117-iq3_s.json`, restart |
| window off, same format | `Environment=STRATA_KV_WINDOW=0` in the unit, restart |
| the whole engine tree | `/root/strata-bak-pre0242-20261025-171737/` (0.1.47), `/root/strata-bak-rows1-20261027-190924/` (0.1.49.2 before the rows1 work), and the oct3-era trees `/root/strata-bak-window-20261030-000411/`, `/root/strata-bak-window2-0247/` |

A rollback that leaves `engine/BUILD.json`'s `version`/`src` fields behind makes later diagnostics wrong; update them
with the binary.

## 7. Pitfalls seen on this box

- `strata-vision` is not part of the main CMake project; build it from `tools/vision` separately if the unit needs it.
- The first long prompt after a cold start can push the system commit charge hard on Windows HIP; on Linux this box
  has no swap, so an OOM kills rather than pages.
- `--kv oct3` is refused by the row-batched and slot-batched attention paths on purpose (they exit with a message).
  Do not enable `STRATA_BATCH_ATTN_ROWS` or a multi-slot (`--replicas`) server with oct3.
- The oct3 window is a cache: a restored conversation slot refills it as it appends. Its first tokens after a restore
  read 3-bit codes for the recent cells, which is why a slot's saved K/V stays valid.
