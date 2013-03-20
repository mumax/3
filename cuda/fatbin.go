package cuda

import "github.com/barnex/cuda5/cu"

// load PTX code for function name, find highest SM that matches our card.
func fatbinLoad(sm map[int]string, fn string) cu.Function {
	best := 0
	for k, _ := range sm {
		if k <= cudaCC && k > best {
			best = k
		}
	}
	return cu.ModuleLoadData(sm[best]).GetFunction(fn)
}
