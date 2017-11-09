package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha        = NewScalarParam("alpha", "", "Landau-Lifshitz damping constant")
	Xi           = NewScalarParam("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol          = NewScalarParam("Pol", "", "Electrical current polarization Zhang-Li")
	PolSL        = NewScalarParam("PolSL", "", "Electrical current polarization Slonczewski")
	Lambda       = NewScalarParam("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime = NewScalarParam("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins  = NewScalarParam("frozenspins", "", "Defines spins that should be fixed") // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values

	FixedLayer                       = NewExcitation("FixedLayer", "", "Slonczewski fixed layer polarization")
	Torque                           = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque                         = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque                         = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	J                                = NewExcitation("J", "A/m2", "Electrical current density in ferromagnet")
	JHM                              = NewExcitation("JHM", "A/m2", "Electrical current density in heavy metal")
	MaxTorque                        = NewScalarValue("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)
	GammaLL                  float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                          = true
	DisableZhangLiTorque             = false
	DisableSlonczewskiTorque         = false
	IndpendentSlonczewskiTorque      = false
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	DeclVar("IndpendentSlonczewskiTorque", &IndpendentSlonczewskiTorque, "Indpendent Slonczewski torque (default=false)")
}

// Sets dst to the current total torque
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



///////////////// For LLB the same previous two functions

// Sets dst to the current total LLBtorque
func SetTorqueLLB(dst *data.Slice,hth1 *data.Slice,hth2 *data.Slice) {
	SetLLBTorque(dst,hth1,hth2)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz-Bloch torque
func SetLLBTorque(dst *data.Slice,hth1 *data.Slice,hth2 *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	Temp := Temp.MSlice()
	defer Temp.Recycle()

        cuda.Zero(hth1)       
        B_therm.AddTo(hth1)
        cuda.Zero(hth2)       
        B_therm.AddTo(hth2)
	if Precess {
		cuda.LLBTorque(dst, M.Buffer(), dst, Temp,alpha,TCurie,Msat,hth1,hth2) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

//////////////////////////////////////////////////////////




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

		if (IndpendentSlonczewskiTorque== false){
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
		}else{
			msat := Msat.MSlice()
			defer msat.Recycle()
			jHM := JHM.MSlice()
			defer jHM.Recycle()
			fixedP := FixedLayer.MSlice()
			defer fixedP.Recycle()
			alpha := Alpha.MSlice()
			defer alpha.Recycle()
			polSL := PolSL.MSlice()
			defer polSL.Recycle()
			lambda := Lambda.MSlice()
			defer lambda.Recycle()
			epsPrime := EpsilonPrime.MSlice()
			defer epsPrime.Recycle()
			cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
				msat, jHM, fixedP, alpha, polSL, lambda, epsPrime, Mesh())
		}
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

func GetMaxTorque() float64 {
	torque := ValueOf(Torque)
	defer cuda.Recycle(torque)
	return cuda.MaxVecNorm(torque)
}
