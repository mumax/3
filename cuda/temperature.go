package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
	"unsafe"
)

func AddTemperature(Beff, noise *data.Slice, Temp, alpha, Bsat LUTPtr, VolDt float64, regions *Bytes) {

	util.Argument(Beff.NComp() == 1 && noise.NComp() == 1)

	N := Beff.Len()
	cfg := make1DConf(N)

	kB2_VgammaDt := mag.Mu0 * mag.Kb / (mag.Gamma0 * VolDt) // TODO -> engine

	k_addtemperature(Beff.DevPtr(0), noise.DevPtr(0), float32(kB2_VgammaDt),
		unsafe.Pointer(Temp), unsafe.Pointer(alpha), unsafe.Pointer(Bsat),
		regions.Ptr, N, cfg)
}
