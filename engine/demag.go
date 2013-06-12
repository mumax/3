package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Msat        ScalarParam                   // Saturation magnetization in A/m
	bsat        ScalarParam                   // automatically derived from Msat
	B_demag     setterQuant                   // demag field in Tesla
	FFTM        fftm                          // FFT of m
	EnableDemag bool                   = true // enable/disable demag field
	demag_      *cuda.DemagConvolution        // does the heavy lifting and provides FFTM
)

func init() {
	world.LValue("Msat", &Msat)
	world.Var("EnableDemag", &EnableDemag)

	fftm_addr := &FFTM
	world.ROnly("mFFT", &fftm_addr)

	B_demag_addr := &B_demag
	world.ROnly("B_demag", &B_demag_addr)
}

func initDemag() {
	demag_ = cuda.NewDemag(Mesh())
	Msat = scalarParam("Msat", "A/m")
	bsat = scalarParam("Bsat", "T")
	Msat.post_update = func(region int) {
		bsat.setRegion(region, Msat.GetRegion(region)*Mu0)
	}
	B_demag = setter(3, Mesh(), "B_demag", "T", func(b *data.Slice, cansave bool) {
		if EnableDemag {
			demag_.Exec(b, M.buffer, bsat.Gpu(), regions.Gpu())
		} else {
			cuda.Zero(b)
		}
	})
	Quants["B_demag"] = &B_demag
}
