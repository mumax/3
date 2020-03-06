package cuda

import (
	"github.com/mumax/3/data"
	"log"
)

func SingleFFT(outReal, outImag, in *data.Slice) {

	if outReal.NComp() != in.NComp() || outImag.NComp() != in.NComp() {
		log.Panicf("number of components do not match for fft input and output")
	}

	inSize := in.Size()

	for i := 0; i < 3; i++ {
		if inSize[i] != outReal.Size()[i] || inSize[i] != outImag.Size()[i] {
			log.Panicf("fft input and output size do not match")
		}
	}

	plan := newFFT3DR2C(inSize[X], inSize[Y], inSize[Z])
	defer plan.Free()

	nxOut, nyOut, nzOut := plan.OutputSizeFloats()
	outSize := [3]int{nxOut, nyOut, nzOut}

	out := Buffer(1, outSize) // re-use output buffer for every component
	defer Recycle(out)

	for c := 0; c < in.NComp(); c++ {
		plan.ExecAsync(in.Comp(c), out.Comp(0))
		fftShift(outReal.Comp(c), outImag.Comp(c), out.Comp(0))
	}

}

// Applies the fft shift on an fft ouput Slice and
// seperates the real and imaginary part
func fftShift(outReal, outImag, fftOutput *data.Slice) {

	N := outReal.Size()
	cfg := make3DConf(N)

	k_fftshift_async(
		outReal.DevPtr(0),
		outImag.DevPtr(0),
		fftOutput.DevPtr(0),
		N[X], N[Y], N[Z], cfg)
}
