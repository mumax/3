package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Alpha, Xi ScalarParam
	LLTorque  setter     // Landau-Lifshitz torque/γ0, in T
	STTorque  adder      // Spin-transfer torque/γ0, in T
	JPol      excitation // Polarized electrical current density
	MaxTorque = NewGetScalar("maxTorque", "T", "", GetMaxTorque)
)

func init() {
	Alpha.init("alpha", "", "Landau-Lifshitz damping constant", nil)
	Xi.init("xi", "", "Non-adiabaticity of spin-transfer-torque", nil)
	JPol.init("JPol", "A/m2", "Polarized electrical current density")

	//DeclROnly("MaxTorque", &MaxTorque, "Maximum total torque (T)")

	LLTorque.init(3, &globalmesh, "lltorque", "T", "Landau-Lifshitz torque/γ0", func(b *data.Slice) {
		B_eff.set(b)
		cuda.LLTorque(b, M.buffer, b, Alpha.LUT1(), regions.Gpu())
	})

	STTorque.init(3, &globalmesh, "sttorque", "T", "Spin-transfer torque/γ0", func(dst *data.Slice) {
		if !JPol.isZero() {
			jspin, rec := JPol.Get()
			if rec {
				defer cuda.Recycle(jspin)
			}
			cuda.AddZhangLiTorque(dst, M.buffer, jspin, Bsat.LUT1(), Alpha.LUT1(), Xi.LUT1(), regions.Gpu())
		}
	})

	Torque.init(3, &globalmesh, "torque", "T", "Total torque/γ0", func(b *data.Slice) {
		LLTorque.set(b)
		STTorque.addTo(b)
	})

}

// TODO: could implement maxnorm(torque) (getfunc)
func GetMaxTorque() float64 {
	torque, recycle := Torque.Get()
	if recycle {
		defer cuda.Recycle(torque)
	}
	return cuda.MaxVecNorm(torque)
}
