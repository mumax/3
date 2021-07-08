package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// dst = sqrt(dot(a, a)),
func VecNorm(dst *data.Slice, a *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_vecnorm_async(dst.DevPtr(0),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		N, cfg)
}
