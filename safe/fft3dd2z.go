package safe

import (
	"fmt"
	"github.com/barnex/cuda4/cufft"
)

// 3D single-precission real-to-complex FFT plan.
type FFT3DD2ZPlan struct {
	fftplan
	size3D
}

// 3D single-precission real-to-complex FFT plan.
func FFT3DD2Z(Nx, Ny, Nz int) FFT3DD2ZPlan {
	handle := cufft.Plan3d(Nx, Ny, Nz, cufft.D2Z)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	return FFT3DD2ZPlan{fftplan{handle, 0}, size3D{Nx, Ny, Nz}}
}

// Execute the FFT plan.
// src and dst are 3D arrays stored 1D arrays.
func (p FFT3DD2ZPlan) Exec(src Float64s, dst Complex128s) {
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecD2Z(src.Pointer(), dst.Pointer())
}

// 3D size of the input array.
func (p FFT3DD2ZPlan) InputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]
}

// 3D size of the output array.
func (p FFT3DD2ZPlan) OutputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]/2 + 1
}

// Required length of the (1D) input array.
func (p FFT3DD2ZPlan) InputLen() int {
	return prod3(p.InputSize())
}

// Required length of the (1D) output array.
func (p FFT3DD2ZPlan) OutputLen() int {
	return prod3(p.OutputSize())
}
