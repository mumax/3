# mumax3-cuda-graph

**CUDA Graph–accelerated fork of [mumax³](https://mumax.github.io) — transparent step-time speedup for small-to-medium micromagnetic simulations.**

This fork adds a CUDA Graph capture/replay path on top of mumax³ (based on commit [`3fe3d41`](https://github.com/mumax/3/commit/3fe3d41)) that transparently accelerates `Steps()` / `Run()` calls without requiring any changes to existing `.mx3` scripts.

---

## Performance

All measurements on **RTX 5060 Ti** (Blackwell, 36 SM), CUDA 13.2, comparing identical `.mx3` scripts with and without the CUDA Graph path.

| Grid size | Cells | Solver | Speedup |
| --------- | ----- | ------ | ------- |
| 64 × 64 × 1 | 4 K | Heun (fixed dt) | **5.1 – 5.4 ×** |
| 64 × 64 × 1 | 4 K | Heun (adaptive) | **3.4 – 3.6 ×** |
| 64 × 64 × 1 | 4 K | RK45DP (adaptive) | **2.75 – 2.79 ×** |
| 256 × 256 × 1 | 65 K | Heun (fixed dt) | **3.3 – 3.5 ×** |
| 256 × 256 × 1 | 65 K | Heun (adaptive) | **2.4 – 2.5 ×** |
| 256 × 256 × 1 | 65 K | RK45DP (adaptive) | **2.37 – 2.38 ×** |
| 512 × 512 × 1 | 262 K | Heun / RK45DP | **1.3 – 1.5 ×** |
| 800 × 800 × 1 | 640 K | RK45DP | **~1.10 ×** |
| 1024 × 1024 × 1 | 1 M | any | ~1.0 × (no effect) |

The speedup is highest for small and medium grids where kernel-launch overhead dominates, and diminishes as the grid grows GPU-compute-bound (above ~800 K cells).

> The additional lazy-MaxTorque + redundant-Sync removal (also included) gives a further **5 – 19 % improvement** on small/medium grids independent of the CUDA Graph path.

---

## How it works

mumax³ executes ~27 CUDA kernel launches per time step on a single stream, each incurring a full driver round-trip (~7 µs on small grids). On a 64 × 64 grid these launch overheads account for **75 % of total step time**.

This fork:

1. **Captures** one complete time-step kernel sequence as a CUDA Graph (one-time, ~0.5 ms).
2. **Replays** the graph every subsequent step via a single `cuGraphLaunch` call.
3. Handles adaptive-dt solvers with a **split-graph** design: the dt-independent `torqueFn` stages are captured; dt-dependent arithmetic and error estimation run outside the graph.
4. Integrates **transparently** into `Steps()` / `Run()` — no script changes required.

### Safety guards

The graph path is automatically skipped (falling back silently to the original path) when any of the following are detected:

- `Temp ≠ 0` (thermal fluctuations — cuRAND state management in captured graph unverified)
- Time-dependent `B_ext` or material/torque parameters (21 parameters checked)
- `AddFieldTerm` custom field terms
- `NoDemagSpins ≠ 0`
- Mesh cell count > 800 000
- Fewer than 20 steps requested
- Solver other than Heun / RK23 / RK45DP / RK56 / BackwardEuler

---

## Download

### Pre-built Windows binary (CUDA 13.2, sm_75 – sm_120)

👉 **[Download the latest release](https://github.com/mirryou-maker/mumax3-cuda-graph/releases/latest)**

The binary was compiled on Windows 11 with CUDA Toolkit 13.2 and supports Turing (sm_75) and later GPUs (Ampere, Ada, Hopper, Blackwell). For Maxwell / Pascal / Volta GPUs, build from source (see below).

Verify after download:

```sh
mumax3.exe -test
```

### Linux / older GPUs

Build from source (see below).

---

## Building from source

### Prerequisites

| Component | Version tested |
| --------- | -------------- |
| Go | ≥ 1.22 |
| CUDA Toolkit | 13.2 (≤ 12.9 also works with minor adjustments) |
| C compiler | MSVC 14.x (Windows) / GCC (Linux) |
| GPU | NVIDIA, compute capability ≥ 7.5 for this build |

### Windows (CUDA 13.2)

Open PowerShell and run:

```powershell
$env:PATH = "C:\<path-to-mingw64>\bin;" + $env:PATH   # MinGW-w64 for cgo
$env:CGO_ENABLED  = "1"
$env:CGO_CFLAGS   = "-IC:/PROGRA~1/NVIDIA~2/CUDA/v13.2/include -w"
$env:CGO_LDFLAGS  = "-LC:/PROGRA~1/NVIDIA~2/CUDA/v13.2/lib/x64"
go build -o mumax3.exe ./cmd/mumax3
```

> **Note (CUDA 13.x only):** `cuda/cu/context.go` already contains the `cuCtxCreate_v2` patch required for CUDA 13.x compatibility.

### Linux

```bash
export CGO_ENABLED=1
make
```

---

## Changes from upstream mumax³

All source modifications relative to `mumax/3@3fe3d41`:

| File(s) | Change |
| ------- | ------ |
| `cuda/cu/graph.go` *(new)* | CUDA Graph Driver API bindings |
| `cuda/init.go` | `stream0` promoted from `const` to `var`; `EnterCaptureMode` / `ExitCaptureMode` helpers |
| `cuda/conv_demag.go` | `SetStream()` method for FFT plan rebinding during capture |
| `cuda/conv_kernmul.go` + `.cu` | Launch-range halved for 2-D kernel-mul (symmetric demag) |
| `cuda/slice.go` | Removed redundant `Sync()` calls around `MemCpyDtoH` |
| `engine/heun.go` | `StepCaptureBody()` — GPU-only step variant without reduction |
| `engine/graphrun.go` *(new)* | `RunGraph(n)`, `tryRunGraph`, safety guards, per-solver graph runners |
| `engine/run.go` | `Run()` / `Steps()` transparently dispatch to `tryRunGraph` |
| `engine/{rk23,rk45dp,rk56,backwardeuler}.go` | Removed non-load-bearing `setMaxTorque()` calls |
| `engine/gui.go` | `maxtorque` display switched to lazy `GetMaxTorque()` |
| `bench/phase0/` *(new)* | Benchmark and regression scripts used during development |

---

## Compatibility

- All 176 mumax³ regression tests pass (`test/` directory, `cwd=test/`).
- Results are **bit-identical** to upstream mumax³ on the same inputs whenever the graph path is active.
- The graph path silently falls back to the standard path for any unsupported configuration.

---

## Upstream

This fork is based on **mumax³** by Arne Vansteenkiste et al.  
Source: <https://github.com/mumax/3>  
Homepage: <https://mumax.github.io>

If you use this fork in published work, please also cite the original mumax³ paper:

> A. Vansteenkiste, J. Leliaert, M. Dvornik, M. Helsen, F. Garcia-Sanchez, and B. Van Waeyenberge,
> "The design and verification of MuMax3," *AIP Advances* **4**, 107133 (2014).
> <https://doi.org/10.1063/1.4899186>

---

## License

Same as upstream mumax³: [GPL-3.0](LICENSE).
