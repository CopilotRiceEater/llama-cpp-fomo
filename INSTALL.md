# Building llama-cpp-fomo on Windows

Primary target: **RTX 5090 (sm_120a) + MSVC + CUDA + Ninja**. Validated end-to-end from a fresh clone (MSVC 19.44 + CUDA 13.0, Ryzen 9 9950X3D, Windows 11).

Also supported: any RTX 30/40-series card — swap `CMAKE_CUDA_ARCHITECTURES` to `86` (Ampere) or `89` (Ada Lovelace). Parmesan's original benchmark data is on RTX 4090, so pre-Blackwell is well-trodden ground. See [Building for other RTX GPUs](#building-for-other-rtx-gpus) below.

---

## Shell note

Command blocks below use **CMD caret (`^`) line continuation**, which works in:
- `x64 Native Tools Command Prompt for VS 2022`
- Plain `cmd.exe` with VS environment activated

If you prefer **PowerShell** (7+ or Developer PowerShell for VS 2022), replace `^` with backtick `` ` `` at line ends. Example:

```powershell
cmake -B build-sm120a -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=120a
```

To activate the VS environment inside a plain PowerShell session:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

(Replace `Community` with `Professional` / `Enterprise` if that's your VS edition. Note: this script changes the working directory — `cd` back to your repo afterwards.)

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
cmake --build build-sm120a --config Release -j 32
```

Typical build time: **8-15 minutes** on Ryzen 9 9950X3D with `-j 32`. CUDA Flash Attention kernels dominate — nvcc stages within them don't parallelize well, so raising `-j` past physical core count has diminishing returns.

**RAM note**: heavy CUDA compile can hit 2-4 GB per nvcc process. On 32 GB systems, drop `-j` to 8-12 to avoid swap thrashing.

Output (Ninja generator — binaries go into `bin/` directly):
```
build-sm120a\bin\
  llama-server.exe
  llama-cli.exe
  llama-bench.exe
  ...
```

The **Visual Studio generator** (`-G "Visual Studio 17 2022"` instead of `"Ninja"`) puts outputs into `build-sm120a\bin\Release\` instead, matching the `--config` flag. Use whichever layout your scripts expect.

---

## Verify

Quick sanity check (paths below assume Ninja output layout — add `Release\` between `bin\` and the exe name if you used the VS generator):

```
build-sm120a\bin\llama-bench.exe -m path\to\small-model.gguf -ngl 99 -p 512 -n 128
```

Or just get the version to confirm CUDA is linked:

```
build-sm120a\bin\llama-server.exe --version
```

Expected output: CUDA device line with your GPU's name + VRAM, compute capability (12.0 on RTX 5090), and MSVC compiler version.

Should show CUDA backend active and produce tg/pp numbers. If it crashes on load, see [Troubleshooting](#troubleshooting).

For a real-world test with this fork's headline features:

```
build-sm120a\bin\llama-server.exe ^
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

Expect ~42 t/s on RTX 5090. See [BENCHMARKS.md](BENCHMARKS.md) for other validated configurations.

---

## Troubleshooting

### Build fails at `src/llama-moe-fused-cold.cpp` with `'__builtin_prefetch' is undefined`

You're on an older checkout. This fork already replaces it with `_mm_prefetch` via a macro guard. If you see this error, you may have merged upstream changes that reverted it — re-apply the patch in `src/llama-moe-fused-cold.cpp`.

### `fatal error C1083: 'cuda_runtime.h': No such file or directory`

This happens when using the Ninja generator (VS generator papers over it via MSBuild auto-integration). Fixed in this fork by `target_include_directories(llama PRIVATE ${CUDAToolkit_INCLUDE_DIRS})` in `src/CMakeLists.txt`. If you see it anyway, confirm your checkout contains that line and that CUDA Toolkit is installed + `CUDA_PATH` env var resolves.

### `error LNK2005: cudaGetExportTable ... already defined` / `LNK1169: multiple symbols`

`cudart.lib` (dynamic) and `cudart_static.lib` (static) are both being linked. This fork's `src/CMakeLists.txt` deliberately adds only the *include dirs*, not a second cudart library. If you see it, you've likely added `target_link_libraries(llama PRIVATE CUDA::cudart)` somewhere — remove it. ggml-cuda is already pulling the correct variant.

### `tests/test-moe-hot-cache.cpp`: `'setenv' / 'unsetenv': identifier not found`

MSVC doesn't provide POSIX env helpers. This fork adds a `_MSC_VER`-guarded `_putenv_s` shim at the top of that file. If you hit this, confirm your checkout has the shim. Alternatively, build without tests: `-DLLAMA_BUILD_TESTS=OFF`.

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

## Building for other RTX GPUs

The fork's merged upstreams (TurboQuant, Parmesan) use general CUDA features without Blackwell-specific kernels, so any RTX 30/40/50-series card builds fine — just change `CMAKE_CUDA_ARCHITECTURES` to match. Parmesan's original benchmark data is on RTX 4090 (sm_89), so pre-Blackwell is actually the hot-cache's *original* validation target.

| GPU family | Cards (examples) | `CMAKE_CUDA_ARCHITECTURES` |
|---|---|---|
| Blackwell | RTX 5070 / 5080 / 5090 | `120a` |
| Ada Lovelace | RTX 4060 / 4070 / 4080 / 4090 | `89` |
| Ampere | RTX 3060 / 3070 / 3080 / 3090 | `86` |
| Multi-arch binary | any of the above | `86;89;120a` (larger binary, runs on all three) |

**Practical notes for non-5090 cards**:
- **VRAM is the real bottleneck, not compute**. The 42 t/s Qwen3.5-122B-A10B headline needs ~26 GB VRAM for `hot-k=128`. On 10-16 GB cards use smaller quants (IQ2_XXS / IQ1_M for 122B, or smaller dense models). On 24 GB (3090/4090), `hot-k=64` range with UD-IQ3_S should land in a workable spot.
- **Flash Attention on sm_86/sm_89**: fully supported by llama.cpp for the quants we test; no fork-level regression.
- **turbo3 KV**: arch-neutral, no Blackwell dependency.
- **MXFP4**: Ada / Ampere don't have the Blackwell-native MXFP4 repack path, so the *combined* MXFP4 + Parmesan incompatibility is a Blackwell-only issue; non-Blackwell cards don't hit it (but also don't get Blackwell's MXFP4 speed benefits).

No separate build instructions — everything else in this file applies, just swap the arch number.

---

## Updating from upstreams

Since this fork is a cherry-pick of three repos, syncing is manual:

1. Pull new commits from `ggml-org/llama.cpp` into a tracking branch, resolve conflicts
2. Re-apply TurboQuant changes (or cherry-pick new turbo-related commits from TheTom fork)
3. Re-apply Parmesan changes (or cherry-pick from ParmesanParty)
4. Re-verify MSVC patches didn't get clobbered
5. Rebuild and benchmark against reference numbers

No automatic update path. Expect each major llama.cpp release to require a day of merge work.
