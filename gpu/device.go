package gpu

import (
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

type Device int

const GPUDevice Device = 0

func (d Device) InitThread() {
	LockCudaThread()
}

func (d Device) StreamCreate() cu.Stream {
	return cu.StreamCreate()
}

func (d Device) Madd(dst nimble.Slice, src1, src2 nimble.Slice, w1, w2 float32) {
	Madd2(dst.Device(), src1.Device(), src2.Device(), w1, w2)
}
