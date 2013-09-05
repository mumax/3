package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

var (
	Msat    ScalarParam // Saturation magnetization in A/m
	M_full  setter      // non-reduced magnetization in A/m
	B_demag setter      // demag field in Tesla
	//E_demag     = NewGetScalar("E_demag", "J", getDemagEnergy)
	EnableDemag = true                          // enable/disable demag field
	bsat        = scalarParam("Bsat", "T", nil) // automatically derived from Msat, never zero
	demagconv_  *cuda.DemagConvolution          // does the heavy lifting and provides FFTM
)

func init() {
	Msat = scalarParam("Msat", "A/m", func(r int) {
		msat := Msat.GetRegion(r)
		bsat.setRegion(r, msat*mag.Mu0)
		ku1_red.setRegion(r, safediv(Ku1.GetRegion(r), msat))
		kc1_red.setRegion(r, safediv(Kc1.GetRegion(r), msat))
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})
	DeclLValue("Msat", &Msat, "Saturation magnetization (A/m)")

	M_full.init(3, &globalmesh, "m_full", "A/m", "Unnormalized magnetization", func(dst *data.Slice) {
		msat, r := Msat.Get()
		if r {
			defer cuda.RecycleBuffer(msat)
		}
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), M.buffer.Comp(c), msat)
		}
	})

	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")

	B_demag.init(3, &globalmesh, "B_demag", "T", "Magnetostatic field (T)", func(b *data.Slice) {
		if EnableDemag {
			demagConv().Exec(b, M.buffer, bsat.Gpu(), regions.Gpu())
		} else {
			cuda.Zero(b)
		}
	})

	registerEnergy(getDemagEnergy)
	//DeclROnly("E_demag", &E_demag, "Magnetostatic energy (J)")
}

func demagConv() *cuda.DemagConvolution {
	if demagconv_ == nil {
		demagconv_ = cuda.NewDemag(Mesh())
	}
	return demagconv_
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
