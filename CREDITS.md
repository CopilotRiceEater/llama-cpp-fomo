# Credits

`llama-cpp-fomo` is a three-way merge built on the shoulders of independent upstream projects. All code from those projects is used under their original licenses (MIT or equivalent).

---

## Primary upstreams

### ggml-org/llama.cpp

- Repo: https://github.com/ggml-org/llama.cpp
- License: MIT
- Maintainers: Georgi Gerganov and contributors
- Role in this fork: **base engine**. Everything else is layered on top of llama.cpp's inference core, model loaders, samplers, and backend abstraction.

### TheTom/llama-cpp-turboquant

- Repo: https://github.com/TheTom/llama-cpp-turboquant
- License: MIT (inherited from llama.cpp)
- Maintainer: TheTom
- Role: **3-bit KV cache types (`turbo2`, `turbo3`, `turbo4`)** based on Walsh-Hadamard transform quantization. Massive KV memory reduction at Q8-competitive quality — critical for running long contexts on 32GB VRAM.
- Integration: cherry-picked cache type definitions, CPU/CUDA kernels, CLI/server plumbing for `--cache-type-k/v turbo[2/3/4]`.

### ParmesanParty/llama.cpp

- Repo: https://github.com/ParmesanParty/llama.cpp
- License: MIT (inherited from llama.cpp)
- Maintainer: ParmesanParty
- Role: **MoE hot-expert cache** — tracks routing frequency and promotes high-frequency experts into VRAM slots while keeping cold experts on CPU. Enables running large MoE models like Qwen3.5-122B-A10B with meaningful speedup over pure CPU offload.
- Integration: cherry-picked `src/llama-moe-hot-cache.{cpp,h}`, `src/llama-moe-fused-cold.{cpp,h}`, graph-builder changes in `src/llama-graph.cpp`, and CLI flags `--moe-hot-k`, `--moe-hot-rebalance-interval`.

---

## This fork's own contributions

All patches below are original work in this fork. Release them upstream freely; we'll help with reproduction if needed.

### MSVC GGUF bool-array parsing fix

- File: `src/llama-model-loader.cpp`
- Symptom: MSVC Release optimization miscompiles `get_arr<bool>`, causing GGUF bool arrays (e.g., Gemma 4's `swa_layers`) to parse as random values.
- Fix: replace the templated path with explicit `uint8_t*` loop.
- Status: not yet reported upstream (as of 2026-04-17).
- Impact without fix: Gemma 4 and similar SWA-using models produce garbage output when built with MSVC Release.

### MSVC prefetch intrinsic compatibility

- File: `src/llama-moe-fused-cold.cpp`
- Symptom: `__builtin_prefetch` is GCC/Clang only; MSVC rejects it.
- Fix: `#if defined(_MSC_VER)` guard that maps it to `_mm_prefetch`.
- Status: local. Could be upstreamed to ParmesanParty.

### Static-link build configuration

- File: `CMakeLists.txt` and build documentation (`INSTALL.md`)
- Symptom: TurboQuant's `turbo3_cpu_wht_group_size` symbol doesn't export cleanly across the `ggml-base` → `ggml-cpu` DLL boundary. Shared-library builds fail at runtime.
- Fix: enforce `BUILD_SHARED_LIBS=OFF` + `GGML_STATIC=ON` in this fork's build path. Documented in `INSTALL.md`.

---

## Models referenced in benchmarks

Models themselves are not redistributed by this fork. Original sources:

- **Qwen3.5-122B-A10B GGUF** (MXFP4 / UD-IQ3_S / etc.): Unsloth's conversions at https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF
- **Gemma 4 31B Dense GGUF**: Unsloth's conversions
- **Qwopus3.5-27B-v3 GGUF**: https://huggingface.co/Jackrong/Qwopus3.5-27B-v3-GGUF

---

## Research references

- TurboQuant (paper/method): see TheTom/llama-cpp-turboquant for the paper link and original authors
- Parmesan hot-cache method: see ParmesanParty/llama.cpp README for the design rationale
- Original llama.cpp CUDA graph work: https://github.com/ggml-org/llama.cpp/issues/6763
- CUDA graph capture fix for volatile prefill: https://github.com/ggml-org/llama.cpp/pull/19754
- `mmvq` for `mul_mat_id` small batch: https://github.com/ggml-org/llama.cpp/pull/18958

---

## License pointer

The LICENSE file at the repo root is the original MIT license from ggml-org/llama.cpp, which covers the entire merged codebase since all upstream projects also use MIT or MIT-compatible licenses.

If you reuse any part of this fork, attribute appropriately to the relevant upstream (llama.cpp / turboquant / parmesan) based on which part you're using. This fork's original patches (MSVC fixes, static-link config, build docs) are also MIT.
