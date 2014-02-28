package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

// Set Bth to thermal noise (Brown).
// see temperature.cu
func SetTemperature(Bth, noise *data.Slice, temp_red data.LUTPtr, kmu0_VgammaDt float64, regions *Bytes) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	k_settemperature_async(Bth.DevPtr(0), noise.DevPtr(0), float32(kmu0_VgammaDt), unsafe.Pointer(temp_red),
		regions.Ptr, N, cfg)
}
