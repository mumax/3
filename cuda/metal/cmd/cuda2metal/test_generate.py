from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("generate.py")
SPEC = importlib.util.spec_from_file_location("cuda2metal_generate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)

REPO_ROOT = SCRIPT.parents[4]
CUDA_DIR = REPO_ROOT / "cuda"
KERNEL_DIR = CUDA_DIR / "metal" / "kernels"
PRELUDE = KERNEL_DIR / "compatibility.metal"


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernels = generator.load_kernels(CUDA_DIR)
        cls.prelude = PRELUDE.read_text(encoding="utf-8")

    def test_audited_kernel_inventory(self) -> None:
        self.assertEqual(
            len(self.kernels), generator.EXPECTED_KERNEL_COUNT
        )
        names = {kernel.name for kernel in self.kernels}
        self.assertEqual(len(names), len(self.kernels))
        self.assertEqual(
            max(len(kernel.arguments) for kernel in self.kernels), 29
        )
        for required in (
            "madd2",
            "addexchange",
            "adddmi",
            "kernmulRSymm3D",
            "reducesum",
            "settemperature2",
            "settopologicalchargelattice",
        ):
            self.assertIn(required, names)

    def test_generated_files_are_current(self) -> None:
        outputs = generator.output_map(
            self.kernels, self.prelude, KERNEL_DIR, CUDA_DIR
        )
        generator.check_outputs(outputs)

    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = Path(first)
            second_path = Path(second)
            outputs_a = generator.output_map(
                self.kernels,
                self.prelude,
                first_path / "kernels",
                first_path / "wrappers",
            )
            outputs_b = generator.output_map(
                self.kernels,
                self.prelude,
                second_path / "kernels",
                second_path / "wrappers",
            )
            contents_a = [
                outputs_a[path] for path in sorted(outputs_a, key=lambda p: p.name)
            ]
            contents_b = [
                outputs_b[path] for path in sorted(outputs_b, key=lambda p: p.name)
            ]
            self.assertEqual(contents_a, contents_b)

    def test_combined_msl_has_no_untranslated_cuda(self) -> None:
        msl = generator.generate_msl(self.kernels, self.prelude)
        for forbidden in (
            'extern "C"',
            "__global__",
            "__device__",
            "__shared__",
            "__syncthreads",
            "#include <stdint.h>",
            "#include <math.h>",
            "#include <stdio.h>",
            "uint8_t",
            "NULL",
            "cuDoubleComplex",
            "make_cuDoubleComplex",
            "cuCadd",
            "cuCsub",
            "cuCmul",
            "cuCimag",
            "atan2f",
        ):
            self.assertNotIn(forbidden, msl)
        self.assertEqual(
            msl.count("kernel void "), generator.EXPECTED_KERNEL_COUNT
        )
        for kernel in self.kernels:
            declaration = f"kernel void {kernel.name}("
            self.assertEqual(msl.count(declaration), 1)
            signature_start = msl.index(declaration)
            signature_end = msl.index("{", signature_start)
            signature = msl[signature_start:signature_end]
            for argument in kernel.arguments:
                self.assertIn(
                    f"[[buffer({argument.buffer_index})]]", signature
                )

    def test_wrapper_preserves_typed_abi(self) -> None:
        source = """extern "C" __global__ void
example(float* __restrict__ output, uint8_t* regions,
        float scale, int count, uint8_t flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
}"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.cu"
            path.write_text(source, encoding="utf-8")
            kernel = generator.parse_kernel(path)
        wrapper = generator.generate_wrapper(kernel)
        for required in (
            "output unsafe.Pointer",
            "regions unsafe.Pointer",
            "scale float32",
            "count int",
            "flags byte",
            "metal.BufferArg(output)",
            "metal.BufferArg(regions)",
            "metal.F32(scale)",
            "metal.I32(count)",
            "metal.U8(flags)",
            "metal.U32(mumaxPointerMask)",
            "kernel example argument output must not be nil",
            "kernel example argument regions must not be nil",
        ):
            self.assertIn(required, wrapper)

    def test_unsupported_signature_fails_closed(self) -> None:
        source = (
            'extern "C" __global__ void bad(double* values, int N) {}'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.cu"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(generator.GenerationError):
                generator.parse_kernel(path)

    def test_file_local_helpers_are_namespaced(self) -> None:
        source = """
__device__ inline float helper(float value) { return value; }
extern "C" __global__ void example(float* dst, int N) {
    if (threadIdx.x < N) { dst[threadIdx.x] = helper(1.0f); }
}"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a-file.cu"
            path.write_text(source, encoding="utf-8")
            kernel = generator.parse_kernel(path)
        translated = generator.translate_kernel(kernel)
        self.assertNotRegex(translated, r"\bhelper\s*\(")
        self.assertIn("helper__a_file", translated)

    def test_manifest_is_public_and_complete(self) -> None:
        document = json.loads(generator.manifest_text(self.kernels))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(
            document["kernel_count"], generator.EXPECTED_KERNEL_COUNT
        )
        encoded = json.dumps(document)
        for forbidden in (
            "original_source",
            "local_macro_names",
        ):
            self.assertNotIn(forbidden, encoded)
        sha256 = re.compile(r"^[0-9a-f]{64}$")
        for kernel in document["kernels"]:
            self.assertRegex(kernel["source_sha256"], sha256)
            self.assertEqual(
                kernel["pointer_mask_buffer_index"],
                len(kernel["arguments"]),
            )

    def test_nullable_pointer_semantics_use_presence_mask(self) -> None:
        msl = generator.generate_msl(self.kernels, self.prelude)
        self.assertNotIn("array == nullptr", msl)
        self.assertIn(
            "amul(vol, mumaxPointerPresent(mumaxPointerMask, 10u)",
            msl,
        )
        self.assertIn(
            "!mumaxPointerPresent(mumaxPointerMask, 3u)? 1.0f: vol[i]",
            msl,
        )
        manifest = json.loads(generator.manifest_text(self.kernels))
        nullable = {
            (kernel["name"], argument["name"])
            for kernel in manifest["kernels"]
            for argument in kernel["arguments"]
            if argument["nullable"]
        }
        self.assertIn(("normalize", "vol"), nullable)
        self.assertIn(("adddmi", "Ms_"), nullable)
        self.assertNotIn(("madd2", "dst"), nullable)
        normalize = next(
            kernel for kernel in self.kernels if kernel.name == "normalize"
        )
        wrapper = generator.generate_wrapper(normalize)
        self.assertIn(
            "kernel normalize argument vx must not be nil", wrapper
        )
        self.assertNotIn(
            "kernel normalize argument vol must not be nil", wrapper
        )


if __name__ == "__main__":
    unittest.main()
