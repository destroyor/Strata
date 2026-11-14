# Strata — RX 7900 XTX Production Fork

A performance-tuned fork of [Strata](https://github.com/Niko1221/Strata) for a **single AMD RX 7900 XTX (gfx1100, RDNA3)** box, optimized **quality-first**: every speed change is validated against the baseline with teacher-forced KL / perplexity / top-1 agreement before it ships.

> 中文说明见下半部分 [中文](#strata--rx-7900-xtx-生产分支)。

---

## What this fork is

This branch (`main`) tracks the **production deployment** exactly. It is the upstream Strata engine plus a small set of measured, quality-gated patches and a documented runtime configuration for one specific machine. It is **not** a general-purpose tuning; the numbers below are for this hardware/model pair and will not transfer unchanged to other setups.

- **Engine**: Strata 0.1.57 (HIP build), base `oct3-v0211` (`29cdbca2`) + the patches below.
- **Production binary sha256**: `1cce3b0dc5b14daa4552dc4809a7a58906a6c3b27602f278a23a0ca21e459aee`.

## Hardware / model target

| Item | Value |
|---|---|
| GPU | AMD RX 7900 XTX, 24 GB, gfx1100 (RDNA3, wave32), 960 GB/s |
| CPU / RAM | Ryzen 9 9950X / 91 GiB |
| ROCm | 10.3.0 |
| Model | Swift-Qwen3.8-Flash-Next GSQ-RCO-abliterated **IQ3_S** + Lamina MTP |
| Context | 262144, **oct3** (3-bit octave) KV |
| Serve | OpenAI-compatible API, port 8089 |

## Patches in this fork

| Commit | Change | Quality |
|---|---|---|
| `5449a12b` | **MMQ ids_identity over-read margin** (cherry-pick `f225844d`): stream-k fixup row table reserves `T*K + roundup(T,J) + J`. Fixes a CUDA Xid 31 FAULT_PDE on layer-split; product values unchanged. | bit-identical |
| `a7a48f04` | **E18 — oct3 decode batched attention**: route `kv_mode==4` through the batched pre75 gather (`STRATA_ATTN_CHUNK_SEL(4, pre)`), opt-in via `STRATA_ATTN_DAILY=1`. Default off keeps the previous kernel byte-for-byte. | top-1 100%, KL 0 |

## Production runtime configuration

Quality-safe optimizations enabled in the systemd user unit (`strata-swift15.service`):

```
STRATA_QSA_ROWS=1        # shuffle-free rows scorer (bit-exact)
STRATA_ATTN_DAILY=1      # E18 batched oct3 attention (top-1 100%, KL 0)
STRATA_SH_STREAM=1       # shared expert on its own stream
STRATA_GDN_SPLIT=1       # GDN step per-head split
STRATA_LFUSE=1           # shared-expert gate+up single launch
STRATA_PLE_BATCH=1       # PLE projection batching
STRATA_ATTN_MERGE_V2=1   # attention merge v2
STRATA_ROUTE_TAIL_SKIP=0 # OFF — reverted for quality (see below)
```

Engine args (key): `--kv oct3 --kv-resident 65536 --spec 4 --spec-min-p 0.2 --ple-io ram --pcie-frac 0 --pool-workers 15 --vram-reserve-mib 576 --conversation-cache-slots 12 --conversation-cache-mib 20480`.

## Quality validation (quality > speed)

Validated with `tools/quant_quality.py` — teacher-forced **top-1 agreement + mean/p99 KL + perplexity** over a fixed text set (code / docs / English / multilingual / agent transcript), production-like expert cache (~8000 slots):

| Switch | top-1 | mean KL | PPL Δ | Decision |
|---|---|---|---|---|
| `ROUTE_TAIL_SKIP=7` | 91.7% | 0.065 | +0.28% | **reverted** (~8% of tokens change argmax; not worth −3.5% speed) |
| `ATTN_DAILY=1` (E18) | **100%** | **0** | 0 | kept |
| `QSA_ROWS=1` / combo winners | bit-exact | 0 | 0 | kept |

The deployed configuration produces **bit-identical output to the baseline** (40K generation sha `876459cb4e` matches the unoptimized baseline). Functional tests alone (standard-v1, needle) are **not** treated as proof of quality.

## Performance

Measured by bracketed A/B on this box (negative = faster), quality-preserving configuration:

- **720K context: ≈ −7–8% ms/window** (QSA_ROWS ≈ −5.2%, E18 ≈ −2.9%, plus the combo winners).
- 160K: ≈ −2–3%; 40K: noise.

## Build / deploy / rollback

```bash
# build (HIP, gfx1100)
cmake -S . -B build-hip -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target strata

# parity gates before deploying
./build-hip/qsa_parity
./build-hip/mmq_ids_margin_test

# deploy: copy build-hip/strata to engine/strata, restart the user unit
systemctl --user restart strata-swift15

# rollback: restore the backed-up binary / unit, restart
```

Backups are kept beside the production binary (`engine/strata.bak-*`) and unit (`strata-swift15.service.bak-*`).

## Branches

- `main` — production state (this document).
- `mmq-fix` — the patch branch `main` was fast-forwarded from.
- `oct3-v0211` — the production base (`29cdbca2`).

---

# Strata — RX 7900 XTX 生产分支

[Strata](https://github.com/Niko1221/Strata) 的性能调优分支，面向**单卡 AMD RX 7900 XTX（gfx1100，RDNA3）**，原则是**质量优先**：每一项提速改动都先用 teacher-forced KL / 困惑度 / top-1 一致率对照基线验证，通过才上线。

## 这个分支是什么

`main` 分支与**生产部署完全一致**：上游 Strata 引擎 + 少量经过实测与质量门禁的补丁 + 针对这台机器的运行配置。它**不是**通用调优，下面的数字只对这套硬件/模型成立，不能原样搬到别的配置。

- **引擎**：Strata 0.1.57（HIP 构建），基线 `oct3-v0211`（`29cdbca2`）+ 下述补丁。
- **生产二进制 sha256**：`1cce3b0dc5b14daa4552dc4809a7a58906a6c3b27602f278a23a0ca21e459aee`。

## 硬件 / 模型

| 项 | 值 |
|---|---|
| GPU | AMD RX 7900 XTX，24 GB，gfx1100（RDNA3，wave32），960 GB/s |
| CPU / 内存 | Ryzen 9 9950X / 91 GiB |
| ROCm | 10.3.0 |
| 模型 | Swift-Qwen3.8-Flash-Next GSQ-RCO-abliterated **IQ3_S** + Lamina MTP |
| 上下文 | 262144，**oct3**（3-bit octave）KV |
| 服务 | OpenAI 兼容 API，端口 8089 |

## 本分支的补丁

| 提交 | 改动 | 质量 |
|---|---|---|
| `5449a12b` | **MMQ ids_identity 过读余量**（cherry-pick `f225844d`）：stream-k fixup 行表预留 `T*K + roundup(T,J) + J`。修复层切分下的 CUDA Xid 31 FAULT_PDE；乘积值不变。 | 逐位一致 |
| `a7a48f04` | **E18 — oct3 解码批量 attention**：把 `kv_mode==4` 走批量 pre75 gather（`STRATA_ATTN_CHUNK_SEL(4, pre)`），`STRATA_ATTN_DAILY=1` opt-in。默认关闭时与原核逐位一致。 | top-1 100%，KL 0 |

## 生产运行配置

systemd user unit（`strata-swift15.service`）中启用的质量保全优化：

```
STRATA_QSA_ROWS=1        # 无 shuffle 的 rows scorer（逐位等价）
STRATA_ATTN_DAILY=1      # E18 批量 oct3 attention（top-1 100%，KL 0）
STRATA_SH_STREAM=1       # shared expert 独立流
STRATA_GDN_SPLIT=1       # GDN step 按 head 切分
STRATA_LFUSE=1           # shared-expert gate+up 单发射
STRATA_PLE_BATCH=1       # PLE 投影合批
STRATA_ATTN_MERGE_V2=1   # attention merge v2
STRATA_ROUTE_TAIL_SKIP=0 # 关闭——因质量回退（见下）
```

关键引擎参数：`--kv oct3 --kv-resident 65536 --spec 4 --spec-min-p 0.2 --ple-io ram --pcie-frac 0 --pool-workers 15 --vram-reserve-mib 576 --conversation-cache-slots 12 --conversation-cache-mib 20480`。

## 质量验证（质量 > 速度）

用 `tools/quant_quality.py` 验证——固定文本集（code / docs / 英文 / 多语言 / agent 转录）上的 teacher-forced **top-1 一致率 + mean/p99 KL + 困惑度**，类生产专家缓存（约 8000 槽）：

| 开关 | top-1 | mean KL | PPL Δ | 判定 |
|---|---|---|---|---|
| `ROUTE_TAIL_SKIP=7` | 91.7% | 0.065 | +0.28% | **回退**（约 8% token 改 argmax，不值 −3.5% 提速） |
| `ATTN_DAILY=1`（E18） | **100%** | **0** | 0 | 保留 |
| `QSA_ROWS=1` / 组合优胜项 | 逐位 | 0 | 0 | 保留 |

部署配置的输出与基线**逐位一致**（40K 生成 sha `876459cb4e` 与未优化基线相同）。功能测试（standard-v1、needle）通过**不**被视为质量无损的证明。

## 性能

本机括号 A/B 实测（负数=更快），质量保全配置：

- **720K 上下文：约 −7–8% ms/窗口**（QSA_ROWS ≈ −5.2%，E18 ≈ −2.9%，加组合优胜项）。
- 160K：约 −2–3%；40K：噪声内。

## 构建 / 部署 / 回滚

```bash
# 构建（HIP，gfx1100）
cmake -S . -B build-hip -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target strata

# 部署前 parity 门禁
./build-hip/qsa_parity
./build-hip/mmq_ids_margin_test

# 部署：把 build-hip/strata 拷到 engine/strata，重启 user unit
systemctl --user restart strata-swift15

# 回滚：恢复备份的二进制 / unit，重启
```

备份保留在生产二进制旁（`engine/strata.bak-*`）与 unit（`strata-swift15.service.bak-*`）。

## 分支

- `main` — 生产状态（本文档）。
- `mmq-fix` — `main` 快进来源的补丁分支。
- `oct3-v0211` — 生产基线（`29cdbca2`）。
