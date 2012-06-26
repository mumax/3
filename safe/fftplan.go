package safe

import (
	"fmt"
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/cufft"
)

type fftplan struct {
	handle cufft.Handle
	cu.Stream
}

type size1D int

func (s size1D) Size() int { return int(s) }

func (p fftplan) Destroy() { p.handle.Destroy() }

func (p fftplan) SetStream(stream cu.Stream) {
	p.handle.SetStream(stream)
	p.Stream = stream
}

type FFT1DR2CPlan struct {
	fftplan
	size1D
	batch int
}

func FFT1DR2C(size, batch int) FFT1DR2CPlan {
	handle := cufft.Plan1d(size, cufft.R2C, batch)
	return FFT1DR2CPlan{fftplan{handle, 0}, size1D(size), batch}
}

func (p FFT1DR2CPlan) Exec(src Float32s, dst Complex64s) {
	if src.Len() != p.batch*p.Size() {
		panic(fmt.Errorf("size mismatch: expecting src len %v, got %v", p.batch*p.Size(), src.Len()))
	}
}
