# CUDA-to-Metal kernel audit

This directory contains the deterministic Metal translation of every
production kernel in `cuda/*.cu`. The inventory is intentionally pinned at 65
kernels. Adding or removing a `.cu` file makes generation fail until the
translator and this audit are reviewed.

## Reproduce

From the repository root:

```sh
python3 cuda/metal/cmd/cuda2metal/generate.py
python3 cuda/metal/cmd/cuda2metal/generate.py --check
python3 -m unittest discover \
  -s cuda/metal/cmd/cuda2metal -p 'test_*.py' -v
```

The generator writes:

- `cuda/*_wrapper_metal.go`: typed Darwin/arm64 launch wrappers.
- `mumax3_kernels.metal`: one self-contained MSL library.
- `manifest.json`: the source hash, kernel name, ordered ABI, nullability, and
  binding index for all 65 kernels.

`source.go` embeds the combined library as `kernels.Source`; the Darwin runtime
must register that value before the first launch.

## Audited CUDA dialect

All production signatures use only:

- `float*` and `uint8_t*` buffers;
- `float`, signed 32-bit `int`, and `uint8_t` scalars;
- one kernel entry point per translation unit.

The translated bodies cover:

- CUDA `blockIdx`, `threadIdx`, `blockDim`, and `gridDim` indexing in 1D and
  3D launch configurations;
- periodic-boundary indexing, clamping, symmetric region lookup tables, and
  byte-valued region maps;
- nullable spatial parameter buffers used by `amul`, `vmul`, `inv_Msat`, and
  `normalize`;
- `float3` vector algebra and the math functions used by the physics kernels;
- interleaved complex-float kernel multiplication;
- all six shared-memory reduction kernels;
- the Hopf-index complex summand.

Any remaining CUDA-only qualifier, warp primitive, CUDA library symbol, or
unsupported signature type causes generation to stop with an error.

## Header mapping

| CUDA header | Metal implementation |
|---|---|
| `amul.h` | Explicit pointer-presence mask plus float/vector multiplier helpers |
| `atomicf.h` | `atomic_uint` compare/exchange preserving CUDA float-bit max semantics |
| `constants.h` | Same constants and expression order |
| `exchange.h` | Same symmetric region-LUT macro |
| `float3.h` | Native MSL `float3`, `dot`, `cross`, and compatibility helpers |
| `reduce.h` | 512-float threadgroup tree with a barrier at every level |
| `stencil.h` | Same flattening, PBC, modulo, and clamp macros |
| `sum.h` | Same two-operand sum |

## ABI and semantic decisions

Each original CUDA argument keeps its ordinal Metal `[[buffer(n)]]` binding.
Buffers are bound with `setBuffer`; scalar values are bound with `setBytes`.
One final `uint32` pointer mask is bound at `buffer(original_argument_count)`.
Bit `n` is set when pointer argument `n` is non-nil. This preserves CUDA's
`NULL`-means-uniform-scalar behavior even though the runtime binds a safe dummy
`MTLBuffer` for nil inputs. Generated wrappers reject nil for every pointer
that is not classified as nullable, preventing an accidental required-buffer
access from being hidden by that dummy allocation.

CUDA's reduction macro assumes implicit warp lockstep below 32 lanes. The Metal
translation does not: it performs a fixed threadgroup tree and executes
`threadgroup_barrier` at every level. Cross-threadgroup accumulation retains
the original atomic behavior and therefore is numerically equivalent within
floating-point reduction-order tolerance, not bitwise deterministic.

The Hopf summand source constructs `cuDoubleComplex` values from float arrays
and writes a float output. Apple GPUs do not expose native FP64, so that
intermediate complex algebra is translated to native `float2`. This is the only
identified precision-width divergence in the 65-kernel corpus and needs a
CUDA/CPU tolerance test on the Hopf observable.

## Coverage

- Element-wise/vector: `cellindices`, `crossproduct`, `pointwise_div`,
  `dotproduct`, `llnoprecess`, `lltorque2`, `madd2` through `madd7`, `minimize`,
  `mul`, `normalize`, `setPhi`, `setTheta`, `settemperature2`.
- Anisotropy/torque/elastic: `addcubicanisotropy2`,
  `adduniaxialanisotropy2`, `addslonczewskitorque2`, `addzhanglitorque2`,
  `addmagnetoelasticfield`, `getmagnetoelasticforce`.
- Exchange, DMI, topology: `addexchange`, `adddmi`, `adddmibulk`,
  `exchangedecode`, `setmaxangle`, all emergent-field and topological-charge
  kernels, `setvectorpotential`.
- FFT support kernels: `copypadmul2`, `copyunpad`, `kernmulC`,
  `kernmulRSymm2Dxy`, `kernmulRSymm2Dz`, `kernmulRSymm3D`,
  `solidanglefourierfield`, `scaleemergentfield`,
  `solidanglefouriersummand`.
- Region/data movement: `crop`, `resize`, all region operations, all byte and
  float shift operations, and both zero-mask operations.
- Reductions: `reducedot`, `reducemaxabs`, `reducemaxdiff`,
  `reducemaxvecdiff2`, `reducemaxvecnorm2`, `reducesum`.

Linux validation checks inventory, ABI typing and order, source hashes,
determinism, helper namespacing, nullable-buffer behavior, checked-in output
freshness, and absence of untranslated CUDA constructs. MSL compilation and
GPU numerical validation still require Apple's Metal compiler and hardware.
