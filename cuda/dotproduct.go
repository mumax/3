package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// dst += v * dot(a, b), as used for energy density
func AddDotProduct(dst, a, b, vol *data.Slice) {
	util.Argument(vol == nil || vol.NComp() == 1)
	util.Argument(dst.NComp() == 1 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_dotproduct_async(dst.DevPtr(0),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		vol.DevPtr(0), N, cfg, stream0)
}
