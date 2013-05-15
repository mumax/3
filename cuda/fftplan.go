package cuda

// INTERNAL
// Base implementation for all FFT plans.

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/cufft"
)

// Base implementation for all FFT plans.
type fftplan struct {
	handle cufft.Handle
	stream cu.Stream
}

func prod3(x, y, z int) int {
	return x * y * z
}

// Releases all resources associated with the FFT plan.
func (p *fftplan) Destroy() { p.handle.Destroy() }

// Associates a CUDA stream with the FFT plan.
func (p *fftplan) setStream(stream cu.Stream) {
	p.handle.SetStream(stream)
	p.stream = stream
}
