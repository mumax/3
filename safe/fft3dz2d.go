package safe

import (
	"fmt"
	"github.com/barnex/cuda5/cufft"
)

// 3D single-precission real-to-complex FFT plan.
type FFT3DZ2DPlan struct {
	fftplan
	size3D
}

// 3D single-precission real-to-complex FFT plan.
func FFT3DZ2D(Nx, Ny, Nz int) FFT3DZ2DPlan {
	handle := cufft.Plan3d(Nx, Ny, Nz, cufft.Z2D)
	handle.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	return FFT3DZ2DPlan{fftplan{handle, 0}, size3D{Nx, Ny, Nz}}
}

// Execute the FFT plan.
// src and dst are 3D arrays stored 1D arrays.
func (p FFT3DZ2DPlan) Exec(src Complex128s, dst Float64s) {
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecZ2D(src.Pointer(), dst.Pointer())
}

// 3D size of the input array.
func (p FFT3DZ2DPlan) InputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]/2 + 1
}

// 3D size of the output array.
func (p FFT3DZ2DPlan) OutputSize() (Nx, Ny, Nz int) {
	return p.size3D[0], p.size3D[1], p.size3D[2]
}

// Required length of the (1D) input array.
func (p FFT3DZ2DPlan) InputLen() int {
	return prod3(p.InputSize())
}

// Required length of the (1D) output array.
func (p FFT3DZ2DPlan) OutputLen() int {
	return prod3(p.OutputSize())
}
