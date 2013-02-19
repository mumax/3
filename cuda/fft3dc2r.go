package cuda

import (
	"code.google.com/p/mx3/data"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/cufft"
)

// 3D single-precission real-to-complex FFT plan.
type FFT3DC2RPlan struct {
	fftplan
	size3D
}

// 3D single-precission real-to-complex FFT plan.
func NewFFT3DC2R(Nx, Ny, Nz int, stream cu.Stream) FFT3DC2RPlan {
	handle := cufft.Plan3d(Nx, Ny, Nz, cufft.C2R)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	handle.SetStream(stream)
	return FFT3DC2RPlan{fftplan{handle, 0}, size3D{Nx, Ny, Nz}}
}

// Execute the FFT plan.
// src and dst are 3D arrays stored 1D arrays.
func (p FFT3DC2RPlan) Exec(src, dst *data.Slice) {
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecC2R(cu.DevicePtr(src.DevPtr(0)), cu.DevicePtr(dst.DevPtr(0)))
	p.stream.Synchronize() //!
}

// 3D size of the input array.
func (p FFT3DC2RPlan) InputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]/2 + 1
}

// 3D size of the output array.
func (p FFT3DC2RPlan) OutputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]
}

// Required length of the (1D) input array.
func (p FFT3DC2RPlan) InputLen() int {
	return prod3(p.InputSize())
}

// Required length of the (1D) output array.
func (p FFT3DC2RPlan) OutputLen() int {
	return prod3(p.OutputSize())
}
