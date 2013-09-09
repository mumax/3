package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Alpha, Xi ScalarParam
	LLTorque  setter     // Landau-Lifshitz torque/γ0, in T
	STTorque  adder      // Spin-transfer torque/γ0, in T
	JPol      excitation // Polarized electrical current density
	MaxTorque = NewGetScalar("maxTorque", "T", "", GetMaxTorque)
)

func init() {
	Alpha.init("alpha", "", "Landau-Lifshitz damping constant")
	Xi.init("xi", "", "Non-adiabaticity of spin-transfer-torque")
	JPol.init("JPol", "A/m2", "Polarized electrical current density")

	//DeclROnly("MaxTorque", &MaxTorque, "Maximum total torque (T)")

	LLTorque.init(3, &globalmesh, "lltorque", "T", "Landau-Lifshitz torque/γ0", func(b *data.Slice) {
		B_eff.set(b)
		cuda.LLTorque(b, M.buffer, b, Alpha.Gpu(), regions.Gpu())
	})

	STTorque.init(3, &globalmesh, "sttorque", "T", "Spin-transfer torque/γ0", func(dst *data.Slice) {
		if !JPol.IsZero() {
			jspin, rec := JPol.Get()
			if rec {
				defer cuda.RecycleBuffer(jspin)
			}
			cuda.AddZhangLiTorque(dst, M.buffer, jspin, bsat.Gpu1(), Alpha.Gpu(), Xi.Gpu(), regions.Gpu())
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
		defer cuda.RecycleBuffer(torque)
	}
	return cuda.MaxVecNorm(torque)
}
