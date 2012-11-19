package uni

import (
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
)

type Device interface {
	InitThread()
	Madd(dst nimble.Slice, src1, src2 nimble.Slice, w1, w2 float32, str cu.Stream)
}
