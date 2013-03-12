package cuda

import "github.com/barnex/cuda5/cu"

func fatbinLoad(sm map[int]string, fn string) cu.Function {
	return cu.ModuleLoadData(sm[20]).GetFunction(fn)
}
