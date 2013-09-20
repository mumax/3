package cuda

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/cufft"
	"github.com/mumax/3/data"
)

// 3D single-precission real-to-complex FFT plan.
type fft3DC2RPlan struct {
	fftplan
	size [3]int
}

// 3D single-precission real-to-complex FFT plan.
func newFFT3DC2R(Nx, Ny, Nz int, stream cu.Stream) fft3DC2RPlan {
	handle := cufft.Plan3d(Nx, Ny, Nz, cufft.C2R)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	handle.SetStream(stream)
	return fft3DC2RPlan{fftplan{handle, 0}, [3]int{Nx, Ny, Nz}}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DC2RPlan) ExecAsync(src, dst *data.Slice) {
	oksrclen := p.InputLenFloats()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLenFloats()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecC2R(cu.DevicePtr(uintptr(src.DevPtr(0))), cu.DevicePtr(uintptr(dst.DevPtr(0))))
}

// Execute the FFT plan, synchronized.
func (p *fft3DC2RPlan) Exec(src, dst *data.Slice) {
	p.ExecAsync(src, dst)
	p.stream.Synchronize()
}

// 3D size of the input array.
func (p *fft3DC2RPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[0], p.size[1], p.size[2] + 2
}

// 3D size of the output array.
func (p *fft3DC2RPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[0], p.size[1], p.size[2]
}

// Required length of the (1D) input array.
func (p *fft3DC2RPlan) InputLenFloats() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DC2RPlan) OutputLenFloats() int {
	return prod3(p.OutputSizeFloats())
}
