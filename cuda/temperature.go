package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set Bth to thermal noise (Brown).
// see temperature2.cu
func SetTemperature(Bth, noise *data.Slice, k2mu0_Mu0VgammaDt float64, Msat, Temp, Alpha MSlice) {

	// N is set by the length of Bth. noise is now set to length N+N%2 (N when N is even, N+1 when N is odd) in engine/temperature.go, so noise[N] is a dummy value that should never be used. The kernel only cares about the underlying pointer and the explicit N count, not the declared shape.
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	k_settemperature2_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_Mu0VgammaDt),
		Msat.DevPtr(0), Msat.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		Alpha.DevPtr(0), Alpha.Mul(0),
		N, cfg)
}
