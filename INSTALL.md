# Building llama-cpp-fomo on Windows

Target: **RTX 5090 (sm_120a) + MSVC + CUDA**. Other configurations probably work but aren't tested.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Visual Studio 2022 | 17.x | Install the **"Desktop development with C++"** workload. MSVC v143 toolset. |
| CUDA Toolkit | 12.6+ | Must support `sm_120a`. CUDA 13.x recommended. |
| NVIDIA Driver | 595+ | Ships with Blackwell support. |
| CMake | 3.24+ | Bundled with VS 2022 or install standalone. |
| Git | 2.40+ | With Git LFS if cloning models separately. |
| Python | 3.10+ | Optional, for conversion / benchmark scripts. |

**Verify before proceeding**:

```
nvcc --version
cl.exe            # inside "x64 Native Tools Command Prompt for VS 2022"
cmake --version
```

`nvcc` should report CUDA 12.6 or newer. `cl.exe` should report MSVC 19.40+ (VS 2022 17.10+).

---

## Clone

```
cd G:\
git clone https://github.com/CopilotRiceEater/llama-cpp-fomo.git
cd llama-cpp-fomo
git submodule update --init --recursive
```

---

## Configure with CMake

From a **"x64 Native Tools Command Prompt for VS 2022"** shell:

```
cmake -B build-sm120a -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DGGML_STATIC=ON ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLAMA_CURL=OFF
```

### Flag explanations

| Flag | Why |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES=120a` | Target Blackwell sm_120a only. Use `120a;89;86` if you also want Ada/Ampere. |
| `GGML_CUDA=ON` | Enable CUDA backend. |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Build Flash Attention kernels for all quant types (needed for IQ3_S + FA). |
| `GGML_STATIC=ON` + `BUILD_SHARED_LIBS=OFF` | **Required**. TurboQuant's symbols don't cross DLL boundaries cleanly. |
| `LLAMA_CURL=OFF` | Avoid pulling in curl dependency on Windows. Remove if you want HTTP-based model pulls. |

### Optional flags

- `-DGGML_CUDA_F16=ON` — FP16 accumulation on some kernels, small speed boost
- `-DLLAMA_BUILD_TESTS=OFF` — skip tests (faster build)
- `-DLLAMA_BUILD_EXAMPLES=ON` — include CLI examples (default ON, useful)

---

## Build

```
cmake --build build-sm120a --config Release -j 16
```

Typical build time: **15-25 minutes** on Ryzen 9 9950X3D. CUDA Flash Attention kernels dominate compile time.

Output:
```
build-sm120a\bin\Release\
  llama-server.exe
  llama-cli.exe
  llama-bench.exe
  ...
```

---

## Verify

Quick sanity check with any small model:

```
build-sm120a\bin\Release\llama-bench.exe -m path\to\small-model.gguf -ngl 99 -p 512 -n 128
```

Should show CUDA backend active and produce tg/pp numbers. If it crashes on load, see [Troubleshooting](#troubleshooting).

For a real-world test with this fork's headline features:

```
build-sm120a\bin\Release\llama-server.exe ^
  --model path\to\Qwen3.5-122B-A10B-UD-IQ3_S.gguf ^
  --n-gpu-layers 99 ^
  --no-mmap ^
  --n-cpu-moe 64 ^
  --flash-attn on ^
  --cache-type-k turbo3 ^
  --cache-type-v turbo3 ^
  --ctx-size 32768 ^
  --moe-hot-k 128 ^
  --moe-hot-rebalance-interval 40 ^
  --host 127.0.0.1 ^
  --port 8080
```

Expect ~42 t/s on RTX 5090.

---

## Troubleshooting

### Build fails at `src/llama-moe-fused-cold.cpp` with `'__builtin_prefetch' is undefined`

You're on an older checkout. This fork already replaces it with `_mm_prefetch` via a macro guard. If you see this error, you may have merged upstream changes that reverted it — re-apply the patch in `src/llama-moe-fused-cold.cpp`.

### Link error: `unresolved external symbol turbo3_cpu_wht_group_size`

You didn't set `BUILD_SHARED_LIBS=OFF`. Re-configure with the flag.

### CUDA `sm_120a` not recognized

CUDA Toolkit too old. Need 12.6+ for proper Blackwell support. `nvcc --list-gpu-arch` should include `sm_120`.

### Model loads but outputs garbage

Most likely causes:
1. **MXFP4 + `--moe-hot-k > 0`**: incompatible. Set `--moe-hot-k 0` or use non-MXFP4 quant.
2. **`swa_layers` parsing bug**: if using a Gemma 4 / SWA model without this fork's MSVC fix, bool array may be corrupted. Confirm by checking server log for "swa layer" counts.
3. **Sampler mismatch**: verify temp/top-p match the model's recommended settings.

### Server refuses to bind (Windows firewall)

`--host 0.0.0.0` is often blocked on Windows. Use `--host 127.0.0.1`.

### VRAM exhausted on load

Check monitor is offloaded to iGPU to free the 5090's full 32GB. For Qwen3.5-122B-A10B IQ3_S, `hot-k=128` needs ~26GB — make sure nothing else is hogging VRAM.

### Slow first request

Parmesan hot-cache is in FILLING mode initially. Takes ~300 decodes to transition to STEADY. First request will look slow; second onward should hit full throughput.

---

## Building for other GPUs

Not officially supported, but if you want to try:

- **Ada (4090, etc.)**: `CMAKE_CUDA_ARCHITECTURES=89`. Parmesan hot-cache mechanism may work but isn't validated here.
- **Ampere (3090)**: `86`. Older than ParmesanParty's main test target (they use 4090). Proceed with caution.
- **Multi-arch binary**: `CMAKE_CUDA_ARCHITECTURES=89;120a` — larger binary but runs on both.

Expect lower tg than the 5090 numbers due to VRAM bandwidth and CUDA core differences.

---

## Updating from upstreams

Since this fork is a cherry-pick of three repos, syncing is manual:

1. Pull new commits from `ggml-org/llama.cpp` into a tracking branch, resolve conflicts
2. Re-apply TurboQuant changes (or cherry-pick new turbo-related commits from TheTom fork)
3. Re-apply Parmesan changes (or cherry-pick from ParmesanParty)
4. Re-verify MSVC patches didn't get clobbered
5. Rebuild and benchmark against reference numbers

No automatic update path. Expect each major llama.cpp release to require a day of merge work.
