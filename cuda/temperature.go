package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set Bth to thermal noise (Brown).
// see temperature.cu
func SetTemperature(Bth, noise *data.Slice, temp_red LUTPtr, k2mu0_VgammaDt float64, regions *Bytes) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	k_settemperature_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_VgammaDt), unsafe.Pointer(temp_red),
		regions.Ptr, N, cfg)
}
