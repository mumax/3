package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/cufft"
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

// Execute the FFT plan. Synchronized.
func (p FFT3DR2CPlan) Exec(src, dst cu.DevicePtr) {
	p.handle.ExecR2C(src, dst)
	p.stream.Synchronize()
}

// 3D size of the input array.
func (p FFT3DR2CPlan) InputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]
}

// 3D size of the output array.
func (p FFT3DR2CPlan) OutputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]/2 + 1
}

// Required length of the (1D) input array.
func (p FFT3DR2CPlan) InputLen() int {
	return prod3(p.InputSize())
}

// Required length of the (1D) output array.
func (p FFT3DR2CPlan) OutputLen() int {
	return prod3(p.OutputSize())
}
