package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha        ScalarParam
	Xi           ScalarParam
	Pol          ScalarParam
	Lambda       ScalarParam
	EpsilonPrime ScalarParam
	FixedLayer   VectorParam
	Torque       setter     // total torque in T
	LLTorque     setter     // Landau-Lifshitz torque/γ0, in T
	STTorque     adder      // Spin-transfer torque/γ0, in T
	J            excitation // Polarized electrical current density
	MaxTorque    *GetScalar
)

func init() {
	Alpha.init("alpha", "", "Landau-Lifshitz damping constant", []derived{&temp_red})
	Xi.init("xi", "", "Non-adiabaticity of spin-transfer-torque", nil)
	J.init("J", "A/m2", "Electrical current density")
	Pol.init("Pol", "", "Electrical current polarization", nil)
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.init("Lambda", "", "Slonczewski Λ parameter", nil)
	EpsilonPrime.init("EpsilonPrime", "", "Slonczewski secondairy STT term ε'", nil)
	FixedLayer.init("FixedLayer", "", "Slonczewski fixed layer polarization")
	LLTorque.init(VECTOR, "LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque.init(VECTOR, "STtorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	Torque.init(VECTOR, "torque", "T", "Total torque/γ0", SetTorque)
	MaxTorque = NewGetScalar("maxTorque", "T", "Maximum torque over all cells", GetMaxTorque)
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	B_eff.Set(dst)                                                      // calc and store B_eff
	cuda.LLTorque(dst, M.Buffer(), dst, Alpha.gpuLUT1(), regions.Gpu()) // overwrite dst with torque
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	// TODO: select, xi is not enough
	cuda.AddZhangLiTorque(dst, M.Buffer(), jspin, Bsat.gpuLUT1(),
		Alpha.gpuLUT1(), Xi.gpuLUT1(), Pol.gpuLUT1(), regions.Gpu(), Mesh())

	if !FixedLayer.isZero() {
		cuda.AddSlonczewskiTorque(dst, M.Buffer(), jspin, FixedLayer.gpuLUT(), Msat.gpuLUT1(),
			Alpha.gpuLUT1(), Pol.gpuLUT1(), Lambda.gpuLUT1(), EpsilonPrime.gpuLUT1(), regions.Gpu(), Mesh())
	}
}

// Gets
func GetMaxTorque() float64 {
	torque, recycle := Torque.Slice()
	if recycle {
		defer cuda.Recycle(torque)
	}
	return cuda.MaxVecNorm(torque)
}
