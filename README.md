# llama-cpp-fomo

**Kitchen-sink llama.cpp build for Windows + RTX 5090 + big MoE models.**

A three-way merge of [llama.cpp](https://github.com/ggml-org/llama.cpp) upstream,
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (3-bit KV cache),
and [ParmesanParty/llama.cpp](https://github.com/ParmesanParty/llama.cpp) (MoE hot-expert cache),
with additional MSVC / Windows / Blackwell (sm_120a) patches that aren't yet in any upstream.

Built because no single source had all the pieces needed to run **Qwen3.5-122B-A10B on a 32GB consumer GPU at ~40 t/s on Windows**. FOMO got the best of me.

---

## Highlight benchmarks

All measured on RTX 5090 (32GB VRAM) + Ryzen 9 9950X3D + 64GB DDR5 + Windows 11. Monitor on iGPU so the 5090's full 32GB is available.

| Model | Quant | Config | ctx | tg (t/s) | VRAM |
|---|---|---|---|---|---|
| Qwen3.5-122B-A10B | UD-IQ3_S | ncmoe=64 + hot-k=128 + turbo3 KV | 32K | **42.28** | 25.9 GB |
| Qwen3.5-122B-A10B | UD-IQ3_S | ncmoe=64, hot-k=0 (baseline) | 32K | 28.91 | ~10 GB |
| Qwen3.5-122B-A10B | MXFP4_MOE | ncmoe=32 + turbo3 KV | 262K | 27.00 | 27.4 GB |
| Gemma 4 31B Dense | UD-Q4_K_XL | turbo3 KV, all on GPU | 262K | 61.5 | 27.7 GB |

The 42 t/s number combines **TurboQuant turbo3 KV** with **Parmesan MoE hot-cache** — a combination that isn't publicly documented anywhere else as of 2026-04.

---

## What's inside

### Merged upstreams

| Source | What it brings |
|---|---|
| `ggml-org/llama.cpp` (≈ff5ef8278) | Core inference engine, model support, sampler chain |
| `TheTom/llama-cpp-turboquant` | `turbo2/3/4` KV cache types (3-bit, WHT-based, Q8-competitive quality) |
| `ParmesanParty/llama.cpp` | `--moe-hot-k N` and `--moe-hot-rebalance-interval N` flags for MoE expert caching |

### Patches local to this fork

| Patch | File | Why |
|---|---|---|
| MSVC bool-array parsing fix in `get_arr<bool>` | `src/llama-model-loader.cpp` | MSVC Release optimization breaks GGUF bool-array parsing; `swa_layers` in Gemma 4 and similar SWA models gets corrupted. Replaced with explicit `uint8_t*` loop. Not yet reported upstream. |
| `__builtin_prefetch` → `_mm_prefetch` | `src/llama-moe-fused-cold.cpp` | Parmesan cold kernel uses GCC-only builtin; MSVC needs the SSE intrinsic. |
| Static-link build path | `CMakeLists.txt` + build scripts | `BUILD_SHARED_LIBS=OFF` required because TurboQuant's `turbo3_cpu_wht_group_size` symbol doesn't export across ggml-base → ggml-cpu DLL boundary. |

---

## Scope and non-scope

### Supported

- **Windows 10/11** with MSVC Visual Studio 2022+
- **CUDA** build for **NVIDIA RTX 5090** (Blackwell, sm_120a)
- **MoE models with CPU offload** (`-n-cpu-moe N`): Qwen3.5-MoE, Gemma 4 MoE, Mixtral-style layouts
- **Dense models with turbo3 KV** (all archs)
- Validated: Qwen3.5-122B-A10B, Gemma 4 31B / 26B-MoE, Qwopus3.5-27B-v3

### Not supported / known broken

- **Linux** — not tested, patches may not even apply cleanly
- **Other GPU architectures** — CMake configured for sm_120a only; change `CMAKE_CUDA_ARCHITECTURES` if building for others, but no guarantee the Parmesan hot-cache path works on pre-Blackwell
- **MXFP4 + Parmesan hot-cache** — Blackwell's native MXFP4 repack is incompatible with Parmesan's raw CPU-byte memcpy into GPU slots. Use MXFP4 OR hot-cache, not both.
- **AMD / Intel GPUs** — Vulkan, ROCm, SYCL not tested

### Philosophy

This fork exists to scratch a very specific itch. Issues for unsupported scenarios will be closed. PRs for specific Windows/5090/MoE improvements are welcome with benchmark data.

---

## Quick orientation

- **Building**: see [INSTALL.md](INSTALL.md). Non-trivial on Windows — read it carefully.
- **Running**: example `.bat` scripts in `scripts/` for validated model configurations.
- **Benchmarks**: see [BENCHMARKS.md](BENCHMARKS.md) for reproducible numbers.
- **Upstream attributions and license details**: see [CREDITS.md](CREDITS.md) and [LICENSE](LICENSE).

---

## Why no pre-built binaries

Binaries would attract support requests for scenarios this fork doesn't actively test. Building from source is the filter — if you can build this, you can also read logs and file a useful bug report.

---

## Related reading

- TurboQuant project: https://github.com/TheTom/llama-cpp-turboquant
- Parmesan project: https://github.com/ParmesanParty/llama.cpp
- Upstream llama.cpp: https://github.com/ggml-org/llama.cpp

---

## License

MIT, inherited from llama.cpp upstream. See [LICENSE](LICENSE). Attributions for merged forks in [CREDITS.md](CREDITS.md).
