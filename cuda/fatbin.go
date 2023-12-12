package cuda

import (
	"log"

	"github.com/mumax/3/v3/cuda/cu"
)

// load PTX code for function name, find highest SM that matches our card.
func fatbinLoad(sm map[int]string, fn string) cu.Function {
	cc := determineCC()
	return cu.ModuleLoadData(sm[cc]).GetFunction(fn)
}

var UseCC = 0

func determineCC() int {
	if UseCC != 0 {
		return UseCC
	}

	for k, _ := range madd2_map {
		if k > UseCC && ccIsOK(k) {
			UseCC = k
		}
	}
	if UseCC == 0 {
		log.Fatalln("\nNo binary for GPU. Your nvidia driver may be out-of-date\n")
	}
	return UseCC
}

// check wheter compute capability cc works
func ccIsOK(cc int) (ok bool) {
	defer func() {
		if err := recover(); err == cu.ERROR_NO_BINARY_FOR_GPU {
			ok = false
		}
	}()
	cu.ModuleLoadData(madd2_map[cc]).GetFunction("madd2")
	return true
}
