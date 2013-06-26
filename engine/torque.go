package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.LValue("alpha", &Alpha)
	world.ROnly("LLtorque", &LLTorque)
	world.ROnly("STTorque", &STTorque)
	world.Var("spinpol", &SpinPol) // todo: suck-up in Jpol
	world.LValue("xi", &Xi)
	world.Var("j", &J)
	world.ROnly("MaxTorque", &MaxTorque)
}

var (
	Alpha     = scalarParam("alpha", "", nil)                              // Damping constant
	LLTorque  setterQuant                                                  // Landau-Lifshitz torque/γ0, in T
	STTorque  adderQuant                                                   // Spin-transfer torque/γ0, in T
	Xi                                        = scalarParam("xi", "", nil) // Non-adiabaticity of spin-transfer-torque // TODO: use beta?
	SpinPol   func() float64                  = Const(1)                   // Spin polarization of electrical current
	J         func() [3]float64               = ConstVector(0, 0, 0)       // Electrical current density
	MaxTorque                                 = newGetScalar("maxTorque", "T", GetMaxTorque)
)

func initLLTorque() {
	LLTorque = setter(3, Mesh(), "lltorque", "T", func(b *data.Slice, cansave bool) {
		B_eff.set(b, cansave)
		cuda.LLTorque(b, M.buffer, b, Alpha.Gpu(), regions.Gpu())
	})
	Quants["lltorque"] = &LLTorque
}

func initSTTorque() {
	STTorque = adder(3, Mesh(), "sttorque", "T", func(dst *data.Slice) {
		j := J()
		if j != [3]float64{0, 0, 0} {
			p := SpinPol()
			jx := j[2] * p
			jy := j[1] * p
			jz := j[0] * p
			cuda.AddZhangLiTorque(dst, M.buffer, [3]float64{jx, jy, jz}, bsat.Gpu(), Alpha.Gpu(), Xi.Gpu(), regions.Gpu())
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
