//go:build hip

package main

import (
	"fmt"

	"github.com/mumax/3/cuda"
)

// gpuInfoLine returns the GPU description printed at startup. The HIP build
// loads one generic amdgcnspirv (SPIR-V) image that the runtime finalizes for
// this device; GPUInfo already names the device's gfx arch.
func gpuInfoLine() string {
	return fmt.Sprintf("GPU info: %s, using generic amdgcnspirv image", cuda.GPUInfo)
}

// goBuildTags are the build tags forwarded to "go run" when executing a .go
// input script, so the script is compiled with the same backend as this
// binary. The HIP build must select the hip backend.
const goBuildTags = "hip"
