package safe

import (
	"github.com/barnex/cuda4/cu"
	"runtime"
	"testing"
)

func TestFloat32sSlice(test *testing.T) {
	runtime.LockOSThread()
	cu.Init(0)
	cu.CtxCreate(cu.CTX_SCHED_AUTO, 0).SetCurrent()

	a := MakeFloat32s(100)
	defer a.Free()

	b := make([]float32, 100)

	if a.Len() != len(b) {
		test.Error("len:", a.Len(), "!=", cap(b))
	}
	if a.Cap() != cap(b) {
		test.Error("cap:", a.Cap(), "!=", cap(b))
	}

	c := a.Slice(20, 30)
	d := b[20:30]

	if c.Len() != len(d) {
		test.Error("sliced len:", c.Len(), "!=", cap(d))
	}
	if c.Cap() != cap(d) {
		test.Error("sliced cap:", c.Cap(), "!=", cap(d))
	}
}
