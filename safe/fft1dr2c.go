package safe

import (
	"fmt"
	"github.com/barnex/cuda5/cufft"
)

// 1D single-precission real-to-complex FFT plan.
type FFT1DR2CPlan struct {
	fftplan
	size1D
	batch int
}

// 1D single-precission real-to-complex FFT plan.
func FFT1DR2C(size, batch int) FFT1DR2CPlan {
	handle := cufft.Plan1d(size, cufft.R2C, batch)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	return FFT1DR2CPlan{fftplan{handle, 0}, size1D(size), batch}
}

// Execute the FFT plan.
func (p FFT1DR2CPlan) Exec(src Float32s, dst Complex64s) {
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecR2C(src.Pointer(), dst.Pointer())
}

// Required length of the input array.
func (p FFT1DR2CPlan) InputLen() int {
	return p.batch * p.Size()
}

// Required length of the output array.
func (p FFT1DR2CPlan) OutputLen() int {
	return p.batch * (p.Size()/2 + 1)
}
