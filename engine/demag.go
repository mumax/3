package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Msat        ScalarParam // Saturation magnetization in A/m
	B_demag     SetterQuant // demag field in Tesla
	FFTM        fftm        // FFT of m
	E_demag     GetFunc     // Magnetostatic energy (J)
	EnableDemag = true      // enable/disable demag field

	bsat   = scalarParam("Bsat", "T", nil) // automatically derived from Msat, never zero
	demag_ *cuda.DemagConvolution          // does the heavy lifting and provides FFTM
)

func init() {
	Msat = scalarParam("Msat", "A/m", func(r int) {
		msat := Msat.GetRegion(r)
		bsat.setRegion(r, msat*Mu0)
		ku1_red.setRegion(r, safediv(Ku1.GetRegion(r), msat))
		kc1_red.setRegion(r, safediv(Kc1.GetRegion(r), msat))
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})

	E_demag = NewGetScalar("E_demag", "J", getDemagEnergy)

	World.Var("EnableDemag", &EnableDemag)
	World.ROnly("mFFT", &FFTM)
	World.ROnly("B_demag", &B_demag)
	World.LValue("Msat", &Msat)
	World.ROnly("E_demag", &E_demag)
}

func initDemag() {
	demag_ = cuda.NewDemag(Mesh())
	B_demag = setter(3, Mesh(), "B_demag", "T", func(b *data.Slice, cansave bool) {
		if EnableDemag {
			demag_.Exec(b, M.buffer, bsat.Gpu(), regions.Gpu())
		} else {
			cuda.Zero(b)
		}
	})
	Quants["B_demag"] = &B_demag
	registerEnergy(getDemagEnergy)
}

// Returns the current demag energy in Joules.
func getDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag) / Mu0
}

func safediv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
