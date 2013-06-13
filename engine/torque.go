package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.LValue("alpha", &Alpha)
	torque_ := &Torque
	world.ROnly("torque", &torque_)
	lltorque_ := &LLTorque
	world.ROnly("LLtorque", &lltorque_)
}

var (
	Alpha            = scalarParam("alpha", "", nil) // Damping constant
	LLTorque, Torque setterQuant                     // torque/gamma0, in Tesla
)

func initTorque() {

	// Landau-Lifshitz torque
	LLTorque = setter(3, Mesh(), "lltorque", "T", func(b *data.Slice, cansave bool) {
		B_eff.set(b, cansave)
		cuda.LLTorque(b, M.buffer, b, Alpha.Gpu(), regions.Gpu())
	})
	Quants["lltorque"] = &LLTorque

	// Total torque
	Torque = setter(3, Mesh(), "torque", "T", func(b *data.Slice, cansave bool) {
		LLTorque.set(b, cansave)
		//STTorque.addTo(b, cansave) TODO
	})
	Quants["torque"] = &Torque
}
