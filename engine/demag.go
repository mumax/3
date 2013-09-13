package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

// demag variables
var (
	Msat            ScalarParam
	bsat            derivedParam
	M_full, B_demag setter
	E_demag         = NewGetScalar("E_demag", "J", "Magnetostatic energy", getDemagEnergy)
	EnableDemag     = true                 // enable/disable demag field
	demagconv_      *cuda.DemagConvolution // does the heavy lifting and provides FFTM
)

func init() {
	Msat.init("Msat", "A/m", "Saturation magnetization", []derived{&bsat, &lex2, &ku1_red, &kc1_red})

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

	bsat.init(1, []updater{&Msat}, func(p *derivedParam) {
		Ms := Msat.CpuLUT()
		for i, ms := range Ms[0] {
			p.cpu_buf[0][i] = mag.Mu0 * ms
		}
	})

	B_demag.init(3, &globalmesh, "B_demag", "T", "Magnetostatic field", func(b *data.Slice) {
		if EnableDemag {
			demagConv().Exec(b, M.buffer, bsat.LUT1(), regions.Gpu())
		} else {
			cuda.Zero(b)
		}
	})

	registerEnergy(getDemagEnergy)
}

// returns demag convolution, making sure it's initialized
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
