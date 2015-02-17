package cuda

import (
	"log"

	"github.com/mumax/3/cuda/cu"
)

// load PTX code for function name, find highest SM that matches our card.
func fatbinLoad(sm map[int]string, fn string) cu.Function {
	defer func() {
		if err := recover(); err == cu.ERROR_NO_BINARY_FOR_GPU {
			log.Fatalln(err, ": most likely your nvidia driver is out-of-date.")
		}
	}()
	best := 0
	for k, _ := range sm {
		if k <= cudaCC && k > best {
			best = k
		}
	}
	if best == 0 {
		log.Fatalln("Unsupported GPU compute capability:", cudaCC)
	}
	//log.Println(fn, best)
	return cu.ModuleLoadData(sm[best]).GetFunction(fn)
}
