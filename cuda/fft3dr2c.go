package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/cufft"
	"log"
)

// 3D single-precission real-to-complex FFT plan.
type FFT3DR2CPlan struct {
	fftplan
	size3D
}

// 3D single-precission real-to-complex FFT plan.
func NewFFT3DR2C(Nx, Ny, Nz int, stream cu.Stream) FFT3DR2CPlan {
	handle := cufft.Plan3d(Nx, Ny, Nz, cufft.R2C)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	handle.SetStream(stream)
	return FFT3DR2CPlan{fftplan{handle, 0}, size3D{Nx, Ny, Nz}}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *FFT3DR2CPlan) ExecAsync(src, dst *data.Slice) {
	util.Argument(src.NComp() == 1 && dst.NComp() == 1)
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		log.Panicf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len())
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		log.Panicf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len())
	}
	p.handle.ExecR2C(cu.DevicePtr(src.DevPtr(0)), cu.DevicePtr(dst.DevPtr(0)))
}

// Execute the FFT plan, synchronized.
func (p *FFT3DR2CPlan) Exec(src, dst *data.Slice) {
	p.ExecAsync(src, dst)
	p.stream.Synchronize()
}

// 3D size of the input array.
func (p *FFT3DR2CPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]
}

// 3D size of the output array.
func (p *FFT3DR2CPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2] + 2
}

// Required length of the (1D) input array.
func (p *FFT3DR2CPlan) InputLen() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *FFT3DR2CPlan) OutputLen() int {
	return prod3(p.OutputSizeFloats())
}
