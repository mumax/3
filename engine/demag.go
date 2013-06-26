package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Msat        ScalarParam                     // Saturation magnetization in A/m
	bsat        = scalarParam("Bsat", "T", nil) // automatically derived from Msat, never zero
	B_demag     setterQuant                     // demag field in Tesla
	FFTM        fftm                            // FFT of m
	EnableDemag = true                          // enable/disable demag field
	demag_      *cuda.DemagConvolution          // does the heavy lifting and provides FFTM
	E_demag     = newGetScalar("E_demag", "J", GetDemagEnergy)
)

// Returns the current demag energy in Joules.
func GetDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag) / Mu0
}

func init() {
	Msat = scalarParam("Msat", "A/m", func(r int) {
		msat := Msat.GetRegion(r)
		bsat.setRegion(r, msat*Mu0)
		ku1_red.setRegion(r, safediv(Ku1.GetRegion(r), msat))
		dmi_red.setRegion(r, safediv(DMI.GetRegion(r), msat))
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})

	world.Var("EnableDemag", &EnableDemag)
	world.ROnly("mFFT", &FFTM)
	world.ROnly("B_demag", &B_demag)
	world.LValue("Msat", &Msat)
	world.ROnly("E_demag", &E_demag)
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
	registerEnergy(GetDemagEnergy)
}
