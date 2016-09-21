package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Sets vector dst to zero where mask != 0.
func ZeroMask(dst *data.Slice, mask MSlice) {
	util.Argument(mask.NComp() == 1)
	N := dst.Len()
	cfg := make1DConf(N)

	for c := 0; c < dst.NComp(); c++ {
		k_zeromask_async(dst.DevPtr(c), mask.DevPtr(0), N, cfg)
	}
}
