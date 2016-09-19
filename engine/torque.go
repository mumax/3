package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha        = NewRegionwiseScalar("alpha", "", "Landau-Lifshitz damping constant")
	Xi           = NewRegionwiseScalar("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol          = NewRegionwiseScalar("Pol", "", "Electrical current polarization")
	Lambda       = NewRegionwiseScalar("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime = NewRegionwiseScalar("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins  = NewRegionwiseScalar("frozenspins", "", "Defines spins that are fixed (=1)")
	FixedLayer   = NewExcitation("FixedLayer", "", "Slonczewski fixed layer polarization")
	Torque       = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque     = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque     = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	J            = NewExcitation("J", "A/m2", "Electrical current density")
	MaxTorque    = NewScalarValue("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)

	Precess                  = true
	DisableZhangLiTorque     = false
	DisableSlonczewskiTorque = false

	GammaLL float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
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
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
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
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	if !DisableZhangLiTorque {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		xi := Xi.MSlice()
		defer xi.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		cuda.AddZhangLiTorque(dst, M.Buffer(), msat, j, alpha, xi, pol, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		fixedP := FixedLayer.MSlice()
		defer fixedP.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		epsPrime := EpsilonPrime.MSlice()
		defer epsPrime.Recycle()
		cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
			msat, j, fixedP, alpha, pol, lambda, epsPrime, Mesh())
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
