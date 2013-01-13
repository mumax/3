package gpu

// This files implements loading and caching of CUDA PTX code.

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/cu"
	"sync"
)

var (
	ptxfuncs = make(map[string]cu.Function)
	ptxlock  sync.Mutex
)

func PTXLoad(function string) cu.Function {
	ptxlock.Lock()
	defer ptxlock.Unlock()

	if ptxfuncs[function] == 0 {
		core.Log("loading PTX code for", function)
		code, ok := ptx.Code[function]
		if !ok {
			panic("ptxload: code not found: " + function)
		}
		mod := cu.ModuleLoadData(code)
		ptxfuncs[function] = mod.GetFunction(function)
	}
	return ptxfuncs[function]
}
