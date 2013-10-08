package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Alpha                ScalarParam
	Xi, Pol              ScalarParam
	Lambda, EpsilonPrime ScalarParam
	FixedLayer           VectorParam
	LLTorque             setter     // Landau-Lifshitz torque/γ0, in T
	STTorque             adder      // Spin-transfer torque/γ0, in T
	J                    excitation // Polarized electrical current density
	MaxTorque            = NewGetScalar("maxTorque", "T", "Maximum torque over all cells", GetMaxTorque)
)

func init() {
	Alpha.init("alpha", "", "Landau-Lifshitz damping constant", []derived{&temp_red})
	Xi.init("xi", "", "Non-adiabaticity of spin-transfer-torque", nil)
	J.init("J", "A/m2", "Electrical current density")
	Pol.init("Pol", "", "Electrical current polarization", nil)
	Lambda.init("Lambda", "", "Slonczewski Λ parameter", nil)
	EpsilonPrime.init("EpsilonPrime", "", "Slonczewski secondairy STT term ε'", nil)
	FixedLayer.init("FixedLayer", "", "Slonczewski fixed layer polarization")

	//DeclROnly("MaxTorque", &MaxTorque, "Maximum total torque (T)")

	LLTorque.init(3, &globalmesh, "LLtorque", "T", "Landau-Lifshitz torque/γ0", func(b *data.Slice) {
		B_eff.set(b)
		cuda.LLTorque(b, M.buffer, b, Alpha.gpuLUT1(), regions.Gpu())
	})

	STTorque.init(3, &globalmesh, "STtorque", "T", "Spin-transfer torque/γ0", func(dst *data.Slice) {
		if !J.isZero() {
			jspin, rec := J.Slice()
			if rec {
				defer cuda.Recycle(jspin)
			}
			cuda.AddZhangLiTorque(dst, M.buffer, jspin, Bsat.gpuLUT1(), Alpha.gpuLUT1(), Xi.gpuLUT1(), Pol.gpuLUT1(), regions.Gpu())
		}
	})

	Torque.init(3, &globalmesh, "torque", "T", "Total torque/γ0", func(b *data.Slice) {
		LLTorque.set(b)
		STTorque.addTo(b)
	})

}

// TODO: could implement maxnorm(torque) (getfunc)
func GetMaxTorque() float64 {
	torque, recycle := Torque.Slice()
	if recycle {
		defer cuda.Recycle(torque)
	}
	return cuda.MaxVecNorm(torque)
}
