package cuda

import (
	"log"

	"github.com/mumax/3/cuda/cu"
)

// load PTX code for function name, find highest SM that matches our card.
func fatbinLoad(sm map[int]string, fn string) cu.Function {
	cc := determineCC()
	return cu.ModuleLoadData(sm[cc]).GetFunction(fn)
}

var ccCache = 0

func determineCC() int {
	if ccCache != 0 {
		return ccCache
	}

	for k, _ := range madd2_map {
		if k > ccCache && okCC(k) {
			ccCache = k
		}
	}
	if ccCache == 0 {
		log.Fatalln("Unsupported GPU compute capability:", ccCache)
	}
	return ccCache
}

func okCC(cc int) (ok bool) {
	defer func() {
		if err := recover(); err == cu.ERROR_NO_BINARY_FOR_GPU {
			ok = false
		}
		log.Println("OK CC", cc, ok)
	}()
	cu.ModuleLoadData(madd2_map[cc]).GetFunction("madd2")
	return true
}
