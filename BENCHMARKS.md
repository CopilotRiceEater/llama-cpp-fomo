# Benchmarks

Reproducible numbers with this fork. All runs on the following baseline unless noted:

- **GPU**: NVIDIA RTX 5090 (Blackwell, 32GB VRAM, PCIe 5.0 x16 active)
- **CPU**: AMD Ryzen 9 9950X3D (16C / 32T)
- **RAM**: 64 GB DDR5-6000
- **OS**: Windows 11
- **Monitor**: offloaded to iGPU so 5090's full 32GB is available
- **Build**: MSVC 2022 + CUDA 13.x, sm_120a, static link, Flash Attention ON

---

## Headline: Qwen3.5-122B-A10B UD-IQ3_S + hot-cache

Model: `unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-IQ3_S.gguf` (43.4 GiB, single file)

Common flags:
```
--n-gpu-layers 99 --no-mmap --n-cpu-moe 64 --flash-attn on
--cache-type-k turbo3 --cache-type-v turbo3
--ctx-size 32768 --reasoning off
--moe-hot-rebalance-interval 40
```

Prompt: `재미있는 이야기 해줘` (Korean, "tell me an interesting story"), thinking disabled, steady-state decode measurement (second request after warmup).

| `--moe-hot-k` | VRAM | tg (t/s) | Parmesan hit_rate | vs hot-k=0 |
|---|---|---|---|---|
| 0 (disabled) | ~10 GB | 28.91 | — | baseline |
| 32 | — | 30.81 | ~48% | +6.6% |
| 64 | 16 GB | 33.36 | ~60% | +15.4% |
| 96 | 21 GB | 37.04 | ~64% | +28.1% |
| **128** | **25.9 GB** | **42.28** | **~72%** | **+46.2%** |

### Observations

- **Linear-ish scaling with hot-k**, not saturating at 96
- hit_rate plateaus at 72% even when `hot-k == n_expert == 128`; suspected edge case in Parmesan STEADY rebalance when K equals expert count (every observation should theoretically hit)
- Per-expert VRAM footprint: ~3.3 MB at IQ3_S (vs ~6.2 MB file-size per expert — slots appear to hold a compact subset)
- **PCIe Gen 4 vs Gen 5** made **zero difference** at hot-k=128 — workload is latency-bound (CPU ↔ GPU sync per layer), not bandwidth-bound
- Under load: CPU ~54%, GPU ~42%, neither saturated → stalls from per-layer synchronization between hot (CUDA) and cold (CPU) paths

---

## Qwen3.5-122B-A10B MXFP4 with turbo3 KV + full-context

Model: `unsloth/Qwen3.5-122B-A10B-GGUF/MXFP4_MOE` (63.6 GiB, 3-split)

**Note**: MXFP4 and Parmesan hot-cache are incompatible on Blackwell (bit-mismatch between native MXFP4 repack and Parmesan's raw CPU-byte copy). Hot-cache disabled for these runs.

Common flags:
```
--n-gpu-layers 99 --no-mmap --n-cpu-moe 32 --flash-attn on
--cache-type-k turbo3 --cache-type-v turbo3
```

| ctx | tg (t/s) | pp (t/s) | VRAM | Notes |
|---|---|---|---|---|
| 16K | 27 | 225 | ~24 GB | standard long-context config |
| 262K | 27 | 250 | 27.4 GB | turbo3 KV kept the cost minimal; tg unchanged vs 16K |

**turbo3 observation**: KV memory scales essentially flat from 16K → 262K because 3-bit quantization makes per-token KV ~0.7 GB at 32K and ~4.8 GB at 262K.

---

## Gemma 4 31B Dense + turbo3 KV + 262K ctx

Model: `Gemma-4-31B-UD-Q4_K_XL` (18.3 GiB)

Fully GPU-resident (no `-n-cpu-moe` needed for dense):

```
--n-gpu-layers 99 --flash-attn on
--cache-type-k turbo3 --cache-type-v turbo3
--ctx-size 262144
```

| Metric | Value |
|---|---|
| tg | **61.5 t/s** |
| pp | 900 t/s |
| VRAM | 27.7 GB |

Demonstrates turbo3 KV viability on a dense 31B model at full 262K context with 4 GB VRAM headroom.

---

## PCIe generation experiment (null result)

One of the more informative non-results. Hypothesis was that PCIe 4.0 was bottlenecking the hybrid hot/cold path at 42 t/s.

| PCIe link | tg (t/s) |
|---|---|
| Gen 4 x16 | 42.28 |
| Gen 5 x16 | 42.28 |

**Conclusion**: with Parmesan's current sync structure, per-layer data transfers are small enough that latency (not bandwidth) dominates. PCIe bandwidth upgrade does not help this workload. Focus for future optimization: CUDA graph capture, reducing cross-backend sync points, not faster interconnect.

---

## Thinking ON vs OFF (Qwen3.5-122B-A10B IQ3_S)

Observation: **Qwen3.5's thinking mode hurts output quality** at IQ3_S, despite similar raw speed.

| Mode | tg (t/s) | Story generation quality |
|---|---|---|
| Thinking ON | 42.28 | "Talking dog" cliché, same joke on repeat, 3 infinite-loop events |
| Thinking OFF | 40.68 | Fresh stories each time (slow mailman, lazy wizard), no loops |

At low-bpw FFN (IQ3_S), thinking mode's long reasoning traces appear to regress toward safer/generic completions. Mode is probably tuned for math/code eval scores, not creative generation. Disabling thinking costs ~5% speed but eliminates repetition and loops.

For this fork's intended use (creative + conversational workloads on 122B MoE), **thinking-off is the recommended default**.

---

## Reproducing these numbers

### Recommended default: thinking off, hot-k=128

```
llama-server ^
  --model Qwen3.5-122B-A10B-UD-IQ3_S.gguf ^
  --n-gpu-layers 99 --no-mmap ^
  --n-cpu-moe 64 --flash-attn on ^
  --cache-type-k turbo3 --cache-type-v turbo3 ^
  --ctx-size 32768 ^
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 ^
  --threads 16 --host 127.0.0.1 --port 8008 ^
  --moe-hot-k 128 --moe-hot-rebalance-interval 40 ^
  --reasoning off ^
  --chat-template-kwargs "{\"enable_thinking\": false}"
```

### Baseline (hot-cache disabled)

Drop `--moe-hot-k` (or set to `0`), change `--n-cpu-moe 64` → `32`, keep everything else.

### Thinking-on variant (for comparison)

Remove `--reasoning off` and `--chat-template-kwargs` lines. Note: expect ~42 t/s but degraded creative output quality at IQ3_S (see "Thinking ON vs OFF" section above).

### Measurement methodology

1. Start server
2. Send warm-up prompt `안녕` (short, ignored for measurement — lets Parmesan complete FILLING→STEADY transition)
3. Send target prompt `재미있는 이야기 해줘`
4. Record tg from server's per-request timing log (`eval time = ... tokens per second` line)
5. For hot-cache effects, wait until log shows `FILLING → STEADY` transition before starting measurement
