package cpu

import (
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

type Device int

const CPUDevice Device = 0

func (d Device) InitThread() {
	// intentionally empty
}

func (d Device) StreamCreate() cu.Stream {
	return 0
}

func (d Device) Madd(dst nimble.Slice, src1, src2 nimble.Slice, w1, w2 float32, str cu.Stream) {
	Madd(dst.Host(), src1.Host(), src2.Host(), w1, w2)
}

func Madd(dst []float32, src1, src2 []float32, w1, w2 float32) {
	for i := range dst {
		dst[i] = w1*src1[i] + w2*src2[i]
	}
}
