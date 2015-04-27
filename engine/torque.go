package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha                    ScalarParam
	Xi                       ScalarParam
	Pol                      ScalarParam
	Lambda                   ScalarParam
	EpsilonPrime             ScalarParam
	FrozenSpins              ScalarParam
	FixedLayer               VectorParam
	Torque                   vSetter    // total torque in T
	LLTorque                 vSetter    // Landau-Lifshitz torque/γ0, in T
	STTorque                 vAdder     // Spin-transfer torque/γ0, in T
	J                        excitation // Polarized electrical current density
	MaxTorque                *GetScalar
	GammaLL                  float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                          = true
	DisableZhangLiTorque             = false
	DisableSlonczewskiTorque         = false
)

func init() {
	Alpha.init("alpha", "", "Landau-Lifshitz damping constant", []derived{&temp_red})
	Xi.init("xi", "", "Non-adiabaticity of spin-transfer-torque", nil)
	J.init("J", "A/m2", "Electrical current density")
	Pol.init("Pol", "", "Electrical current polarization", nil)
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.init("Lambda", "", "Slonczewski Λ parameter", nil)
	Lambda.Set(1) // sensible default value (?). TODO: should not be zero
	EpsilonPrime.init("EpsilonPrime", "", "Slonczewski secondairy STT term ε'", nil)
	FrozenSpins.init("frozenspins", "", "Defines spins that should be fixed", nil) // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
	FixedLayer.init("FixedLayer", "", "Slonczewski fixed layer polarization")
	LLTorque.init("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque.init("STtorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	Torque.init("torque", "T", "Total torque/γ0", SetTorque)
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	MaxTorque = NewGetScalar("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	B_eff.Set(dst) // calc and store B_eff
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, Alpha.gpuLUT1(), regions.Gpu()) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
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
	if !DisableZhangLiTorque {
		cuda.AddZhangLiTorque(dst, M.Buffer(), jspin, Bsat.gpuLUT1(),
			Alpha.gpuLUT1(), Xi.gpuLUT1(), Pol.gpuLUT1(), regions.Gpu(), Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		cuda.AddSlonczewskiTorque(dst, M.Buffer(), jspin, FixedLayer.gpuLUT(), Msat.gpuLUT1(),
			Alpha.gpuLUT1(), Pol.gpuLUT1(), Lambda.gpuLUT1(), EpsilonPrime.gpuLUT1(), regions.Gpu(), Mesh())
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
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
