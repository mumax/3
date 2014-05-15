package cuda

// INTERNAL
// Base implementation for all FFT plans.

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/cufft"
)

// Base implementation for all FFT plans.
type fftplan struct {
	handle cufft.Handle
}

func prod3(x, y, z int) int {
	return x * y * z
}

// Releases all resources associated with the FFT plan.
func (p *fftplan) Free() {
	if p.handle != 0 {
		p.handle.Destroy()
		p.handle = 0
	}
}

// Associates a CUDA stream with the FFT plan.
func (p *fftplan) setStream(stream cu.Stream) {
	p.handle.SetStream(stream)
}
