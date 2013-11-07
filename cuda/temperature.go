package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

func AddTemperature(Beff, noise *data.Slice, temp_red LUTPtr, kmu0_VgammaDt float64, regions *Bytes) {

	util.Argument(Beff.NComp() == 1 && noise.NComp() == 1)

	N := Beff.Len()
	cfg := make1DConf(N)

	k_addtemperature_async(Beff.DevPtr(0), noise.DevPtr(0), float32(kmu0_VgammaDt), unsafe.Pointer(temp_red),
		regions.Ptr, N, cfg, stream0)
}
