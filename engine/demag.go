package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	B_demag     setterQuant        // demag field (T) output handle
	FFTM        fftm               // FFT of M
	EnableDemag bool        = true // enable/disable demag field
	demag_      *cuda.DemagConvolution
)

func init() {
	world.Var("EnableDemag", &EnableDemag)
	fftm_addr := &FFTM
	world.ROnly("mFFT", &fftm_addr)
	B_demag_addr := &B_demag
	world.ROnly("B_demag", &B_demag_addr)

}

func initDemag() {
	demag_ = cuda.NewDemag(Mesh())
	B_demag = setter(3, Mesh(), "B_demag", "T", func(b *data.Slice, cansave bool) {
		if EnableDemag {
			sanitycheck()
			demag_.Exec(b, M.buffer, nil, Mu0*Msat()) // vol = nil
		} else {
			cuda.Zero(b)
		}
	})
	Quants["B_demag"] = &B_demag
}
