package safe

import (
	"github.com/barnex/cuda4/cu"
	"reflect"
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

	e := a.Slice(0, 50)
	f := b[0:50]

	if e.Len() != len(f) {
		test.Error("sliced len:", e.Len(), "!=", cap(f))
	}
	if e.Cap() != cap(f) {
		test.Error("sliced cap:", e.Cap(), "!=", cap(f))
	}
}

func TestFloat32sPanic1(test *testing.T) {
	runtime.LockOSThread()
	cu.Init(0)
	cu.CtxCreate(cu.CTX_SCHED_AUTO, 0).SetCurrent()

	defer func() {
		err := recover()
		test.Log("recovered:", err)
		if err == nil {
			test.Fail()
		}
	}()

	a := MakeFloat32s(100)
	defer a.Free()

	a.Slice(-1, 10)
}

func TestFloat32sPanic2(test *testing.T) {
	runtime.LockOSThread()
	cu.Init(0)
	cu.CtxCreate(cu.CTX_SCHED_AUTO, 0).SetCurrent()

	defer func() {
		err := recover()
		test.Log("recovered:", err)
		if err == nil {
			test.Fail()
		}
	}()

	a := MakeFloat32s(100)
	defer a.Free()

	a.Slice(0, 101)
}

func TestFloat32sCopy(test *testing.T) {
	runtime.LockOSThread()
	cu.Init(0)
	cu.CtxCreate(cu.CTX_SCHED_AUTO, 0).SetCurrent()

	a := make([]float32, 100)

	b := MakeFloat32s(100)
	defer b.Free()

	c := MakeFloat32s(100)
	defer c.Free()

	d := make([]float32, 100)

	for i := range a {
		a[i] = float32(i)
	}

	b.CopyHtoD(a)

	c.CopyDtoD(b)

	c.CopyDtoH(d)

	if !reflect.DeepEqual(a, d) {
		test.Error(d)
	}
}
