package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Alpha    = scalarParam("alpha", "", nil) // Damping constant
	LLTorque setter                          // Landau-Lifshitz torque/γ0, in T
	STTorque adder                           // Spin-transfer torque/γ0, in T
	Xi       = scalarParam("xi", "", nil)    // Non-adiabaticity of spin-transfer-torque // TODO: use beta?
	JPol     excitation                      // Polarized electrical current density
	//MaxTorque = NewGetScalar("maxTorque", "T", GetMaxTorque)
)

func init() {
	DeclLValue("alpha", &Alpha, "Landau-Lifshitz damping constant")
	DeclLValue("xi", &Xi, "Non-adiabaticity of spin-transfer-torque")
	DeclLValue("jpol", &JPol, "Polarized electrical current density (A/m²)")
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
			cuda.AddZhangLiTorque(dst, M.buffer, jspin, bsat.Gpu(), Alpha.Gpu(), Xi.Gpu(), regions.Gpu())
		}
	})

	Torque.init(3, &globalmesh, "torque", "T", "Total torque/γ0", func(b *data.Slice) {
		LLTorque.set(b)
		STTorque.addTo(b)
	})

	JPol.init(&globalmesh, "JPol", "A/m2")
}

// TODO: could implement maxnorm(torque) (getfunc)
func GetMaxTorque() float64 {
	torque, recycle := Torque.Get()
	if recycle {
		defer cuda.RecycleBuffer(torque)
	}
	return cuda.MaxVecNorm(torque)
}
