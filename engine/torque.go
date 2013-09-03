package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	DeclLValue("alpha", &Alpha, "Landau-Lifshitz damping constant")
	DeclROnly("LLtorque", &LLTorque, "Landau-Lifshitz torque/γ0 (T)")
	DeclROnly("STTorque", &STTorque, "Spin-transfer torque/γ0 (T)")
	DeclROnly("torque", &Torque, `Total torque/γ0 (T)`)
	DeclLValue("xi", &Xi, "Non-adiabaticity of spin-transfer-torque")
	DeclLValue("jpol", &JPol, "Polarized electrical current density (A/m²)")
	//DeclROnly("MaxTorque", &MaxTorque, "Maximum total torque (T)")
}

var (
	Alpha    = scalarParam("alpha", "", nil) // Damping constant
	LLTorque setterQuant                     // Landau-Lifshitz torque/γ0, in T
	STTorque adderQuant                      // Spin-transfer torque/γ0, in T
	Xi       = scalarParam("xi", "", nil)    // Non-adiabaticity of spin-transfer-torque // TODO: use beta?
	JPol     excitation                      // Polarized electrical current density
	//MaxTorque = NewGetScalar("maxTorque", "T", GetMaxTorque)
)

func initLLTorque() {
	LLTorque = setter(3, Mesh(), "lltorque", "T", func(b *data.Slice) {
		B_eff.set(b)
		cuda.LLTorque(b, M.buffer, b, Alpha.Gpu(), regions.Gpu())
	})
	JPol.init(Mesh(), "JPol", "A/m2")
}

func initSTTorque() {
	STTorque = adder(3, Mesh(), "sttorque", "T", func(dst *data.Slice) {
		if !JPol.IsZero() {
			jspin, rec := JPol.Get()
			if rec {
				defer cuda.RecycleBuffer(jspin)
			}
			cuda.AddZhangLiTorque(dst, M.buffer, jspin, bsat.Gpu(), Alpha.Gpu(), Xi.Gpu(), regions.Gpu())
		}
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
