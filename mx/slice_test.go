package mx

import "testing"

func TestSlice(t *testing.T) {
	LockCudaThread()
	N := 100

	for _, constructor := range []func(int, int) *Slice{NewSlice, NewUnifiedSlice} {
		a := constructor(3, N)
		defer a.Free()
		a.Memset(1, 2, 3)
		Log(a)

		if a.GPUAccess() == false {
			t.Fail()
		}
		if a.Len() != N {
			t.Fail()
		}
		if a.NComp() != 3 {
			t.Fail()
		}

		b := a.Comp(1)
		if b.GPUAccess() == false {
			t.Error("b.GPUAccess", b.GPUAccess())
		}
		if b.Len() != N {
			t.Error("b.Len", b.Len())
		}
		if b.NComp() != 1 {
			t.Error("b.NComp", b.NComp())
		}
	}
}

func TestSliceFree(t *testing.T) {
	LockCudaThread()
	length := 128 * 1024 * 1024
	N := 17
	// not freeing would attempt to allocate 17GB.
	for i := 0; i < N; i++ {
		a := NewSlice(2, length)
		a.Free()
	}
	a := NewSlice(2, length)
	a.Free()
	a.Free() // test double-free
}
