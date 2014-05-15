package cuda

import (
	"github.com/mumax/3/cuda/cu"
	"log"
)

// load PTX code for function name, find highest SM that matches our card.
func fatbinLoad(sm map[int]string, fn string) cu.Function {
	best := 0
	for k, _ := range sm {
		if k <= cudaCC && k > best {
			best = k
		}
	}
	if best == 0 {
		log.Fatalln("Unsupported GPU compute capability:", cudaCC)
	}
	return cu.ModuleLoadData(sm[best]).GetFunction(fn)
}
