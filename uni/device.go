package uni

import (
	"code.google.com/p/mx3/nimble"
)

type Device interface {
	InitThread()
	Madd(dst nimble.Slice, src1, src2 nimble.Slice, w1, w2 float32)
}
