package gpu

// This files implements loading and caching of CUDA PTX code.

import (
	"code.google.com/p/nimble-cube/core"
	"github.com/barnex/cuda5/cu"
)

var ptxfuncs = make(map[string]cu.Function)

func PTXLoad(code, function string) cu.Function {
	if ptxfuncs[code] == 0 {
		core.Log("loading PTX code for", function)
		mod := cu.ModuleLoadData(code)
		ptxfuncs[code] = mod.GetFunction(function)
	}
	return ptxfuncs[code]
}
