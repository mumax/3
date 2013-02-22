package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

// Default accuracy setting for demag kernel.
const DEFAULT_KERNEL_ACC = 6

func NewDemag(mesh *data.Mesh) *DemagConvolution {
	k := mag.BruteKernel(mesh, DEFAULT_KERNEL_ACC)
	return NewConvolution(mesh, k)
}
