package gpu

// This files implements loading and caching of CUDA PTX code.

import (
	"code.google.com/p/nimble-cube/gpu/ptx"
	"github.com/barnex/cuda5/cu"
)

var ptxfuncs = make(map[string]cu.Function)

func PTXLoad(code, function string) cu.Function {
	if ptxfuncs[code] == 0 {
		mod := cu.ModuleLoadData(ptx.EXCHANGE6)
		ptxfuncs[code] = mod.GetFunction(function)
	}
	return ptxfuncs[code]
}
