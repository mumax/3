package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.LValue("alpha", &Alpha)
	world.ROnly("LLtorque", &LLTorque)
	world.ROnly("STTorque", &STTorque)
	world.LValue("xi", &Xi)
	world.LValue("jpol", &JPol)
	world.ROnly("MaxTorque", &MaxTorque)
}

var (
	Alpha     = scalarParam("alpha", "", nil) // Damping constant
	LLTorque  setterQuant                     // Landau-Lifshitz torque/γ0, in T
	STTorque  adderQuant                      // Spin-transfer torque/γ0, in T
	Xi        = scalarParam("xi", "", nil)    // Non-adiabaticity of spin-transfer-torque // TODO: use beta?
	JPol      excitation                      // Polarized electrical current density
	MaxTorque = newGetScalar("maxTorque", "T", GetMaxTorque)
)

func initLLTorque() {
	LLTorque = setter(3, Mesh(), "lltorque", "T", func(b *data.Slice, cansave bool) {
		B_eff.set(b, cansave)
		cuda.LLTorque(b, M.buffer, b, Alpha.Gpu(), regions.Gpu())
	})
	JPol.init(Mesh(), "JPol", "A/m2")
	Quants["lltorque"] = &LLTorque
	Quants["jpol"] = &JPol
}

func initSTTorque() {
	STTorque = adder(3, Mesh(), "sttorque", "T", func(dst *data.Slice) {
		if !JPol.IsZero() {
			jspin, rec := JPol.GetGPU()
			if rec {
				defer cuda.RecycleBuffer(jspin)
			}
			cuda.AddZhangLiTorque(dst, M.buffer, jspin, bsat.Gpu(), Alpha.Gpu(), Xi.Gpu(), regions.Gpu())
		}
	})
	Quants["sttorque"] = &STTorque
}

// TODO: could implement maxnorm(torque) (getfunc)
func GetMaxTorque() float64 {
	torque, recycle := Torque.GetGPU()
	if recycle {
		defer cuda.RecycleBuffer(torque)
	}
	return cuda.MaxVecNorm(torque)
}
