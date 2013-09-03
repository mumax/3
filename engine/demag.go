package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Msat    ScalarParam // Saturation magnetization in A/m
	B_demag setterQuant // demag field in Tesla
	FFTM    fftm        // FFT of m
	//E_demag     = NewGetScalar("E_demag", "J", getDemagEnergy)
	EnableDemag = true                          // enable/disable demag field
	bsat        = scalarParam("Bsat", "T", nil) // automatically derived from Msat, never zero
	demag_      *cuda.DemagConvolution          // does the heavy lifting and provides FFTM
)

func init() {
	Msat = scalarParam("Msat", "A/m", func(r int) {
		msat := Msat.GetRegion(r)
		bsat.setRegion(r, msat*Mu0)
		ku1_red.setRegion(r, safediv(Ku1.GetRegion(r), msat))
		kc1_red.setRegion(r, safediv(Kc1.GetRegion(r), msat))
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})

	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	DeclROnly("mFFT", &FFTM, "Fourier-transformed magnetization")
	DeclROnly("B_demag", &B_demag, "Magnetostatic field (T)")
	DeclLValue("Msat", &Msat, "Saturation magnetization (A/m)")
	//DeclROnly("E_demag", &E_demag, "Magnetostatic energy (J)")
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
	registerEnergy(getDemagEnergy)
}

// Returns the current demag energy in Joules.
func getDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}

func safediv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
