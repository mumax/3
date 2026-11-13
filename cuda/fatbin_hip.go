//go:build hip

package cuda

import (
	"encoding/base64"

	"github.com/mumax/3/cuda/cu"
)

// Load the generic amdgcnspirv (SPIR-V) image for function name. The image is a
// forward-compatible virtual ISA, so hipModuleLoadData finalizes it for the
// present GPU at load time; there is no per-arch image to select.
func fatbinLoad(image []byte, fn string) cu.Function {
	return cu.ModuleLoadData(image).GetFunction(fn)
}

func mustDecodeCodeobj(s string) []byte {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}
