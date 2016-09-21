package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha        = NewScalarInput("alpha", "", "Landau-Lifshitz damping constant")
	Xi           = NewScalarInput("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol          = NewScalarInput("Pol", "", "Electrical current polarization")
	Lambda       = NewScalarInput("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime = NewScalarInput("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins  = NewScalarInput("frozenspins", "", "Defines spins that are fixed (=1)")
	FixedLayer   = NewVectorInput("FixedLayer", "", "Slonczewski fixed layer polarization")
	Torque       = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque     = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque     = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	J            = NewVectorInput("J", "A/m2", "Electrical current density")
	MaxTorque    = NewScalarValue("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)

	Precess                  = true
	DisableZhangLiTorque     = false
	DisableSlonczewskiTorque = false

	GammaLL float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
)

func init() {
	Pol.Set(1)    // default spin polarization
	Lambda.Set(1) // sensible default value (?).
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
}

// Sets dst to the current total torque
// TODO: extensible
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := MSliceOf(Alpha)
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if IsZero(J) {
		return
	}
	util.AssertMsg(!IsZero(Pol), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	if !DisableZhangLiTorque {
		msat := MSliceOf(Msat)
		defer msat.Recycle()
		j := MSliceOf(J)
		defer j.Recycle()
		alpha := MSliceOf(Alpha)
		defer alpha.Recycle()
		xi := MSliceOf(Xi)
		defer xi.Recycle()
		pol := MSliceOf(Pol)
		defer pol.Recycle()
		cuda.AddZhangLiTorque(dst, M.Buffer(), msat, j, alpha, xi, pol, Mesh())
	}
	if !DisableSlonczewskiTorque && !IsZero(FixedLayer) {
		msat := MSliceOf(Msat)
		defer msat.Recycle()
		j := MSliceOf(J)
		defer j.Recycle()
		fixedP := MSliceOf(FixedLayer)
		defer fixedP.Recycle()
		alpha := MSliceOf(Alpha)
		defer alpha.Recycle()
		pol := MSliceOf(Pol)
		defer pol.Recycle()
		lambda := MSliceOf(Lambda)
		defer lambda.Recycle()
		epsPrime := MSliceOf(EpsilonPrime)
		defer epsPrime.Recycle()
		cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
			msat, j, fixedP, alpha, pol, lambda, epsPrime, Mesh())
	}
}

func FreezeSpins(torque *data.Slice) {
	if !IsZero(FrozenSpins) {
		mask := MSliceOf(FrozenSpins)
		defer mask.Recycle()
		cuda.ZeroMask(torque, mask)
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
