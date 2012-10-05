package safe

import (
	"fmt"
	"github.com/barnex/cuda5/cufft"
)

// 1D single-precission complex-to-real FFT plan.
type FFT1DC2RPlan struct {
	fftplan
	size1D
	batch int
}

// 1D single-precission complex-to-real FFT plan.
func FFT1DC2R(size, batch int) FFT1DC2RPlan {
	handle := cufft.Plan1d(size, cufft.C2R, batch)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	return FFT1DC2RPlan{fftplan{handle, 0}, size1D(size), batch}
}

// Execute the FFT plan.
func (p FFT1DC2RPlan) Exec(src Complex64s, dst Float32s) {
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecC2R(src.Pointer(), dst.Pointer())
}

// Required length of the input array.
func (p FFT1DC2RPlan) OutputLen() int {
	return p.batch * p.Size()
}

// Required length of the output array.
func (p FFT1DC2RPlan) InputLen() int {
	return p.batch * (p.Size()/2 + 1)
}
