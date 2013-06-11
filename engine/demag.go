package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Msat        ScalarParam                   // Saturation magnetization in A/m
	B_demag     setterQuant                   // demag field in Tesla
	FFTM        fftm                          // FFT of m
	EnableDemag bool                   = true // enable/disable demag field
	demag_      *cuda.DemagConvolution        // does the heavy lifting and provides FFTM
)

func init() {
	world.Var("EnableDemag", &EnableDemag)
	fftm_addr := &FFTM
	world.ROnly("mFFT", &fftm_addr)
	B_demag_addr := &B_demag
	world.ROnly("B_demag", &B_demag_addr)
	world.LValue("Msat", &Msat)
}

func initDemag() {
	demag_ = cuda.NewDemag(Mesh())
	B_demag = setter(3, Mesh(), "B_demag", "T", func(b *data.Slice, cansave bool) {
		if EnableDemag {
			panic("regions here please")
			demag_.Exec(b, M.buffer, nil, Mu0*Msat.GetUniform()) // vol = nil
		} else {
			cuda.Zero(b)
		}
	})
	Quants["B_demag"] = &B_demag
}
