package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

// Default accuracy setting for demag kernel.
const DEFAULT_KERNEL_ACC = 6

func NewDemag(m *data.Quant) *DemagConvolution {
	k := mag.BruteKernel(m.Mesh(), DEFAULT_KERNEL_ACC)
	return NewConvolution(m, k)
}
