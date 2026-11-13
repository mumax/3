//go:build !hip

package main

import (
	"fmt"

	"github.com/mumax/3/cuda"
)

// gpuInfoLine returns the GPU description printed at startup. The CUDA build
// reports the compute capability of the PTX selected for this device.
func gpuInfoLine() string {
	return fmt.Sprintf("GPU info: %s, using cc=%d PTX", cuda.GPUInfo, cuda.UseCC)
}

// goBuildTags are the build tags forwarded to "go run" when executing a .go
// input script, so the script is compiled with the same backend as this
// binary. The default (CUDA) build needs none.
const goBuildTags = ""
