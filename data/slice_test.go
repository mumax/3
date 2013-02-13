package data

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

func TestSliceHost(t *testing.T) {
	LockCudaThread()
	length := 100
	a := NewUnifiedSlice(3, length)
	defer a.Free()

	b := a.Host()
	if b[0][0] != 0 || b[1][42] != 0 || b[2][99] != 0 {
		t.Fail()
	}

	a.Memset(1, 2, 3)
	b = a.Host()
	if b[0][0] != 1 || b[1][42] != 2 || b[2][99] != 3 {
		t.Fail()
	}
}

func TestSliceSlice(t *testing.T) {
	LockCudaThread()
	length := 100
	a := NewUnifiedSlice(3, length)
	h := a.Host()
	h[1][21] = 42
	b := a.Slice(20, 30)
	if b.Len() != 30-20 {
		t.Fail()
	}
	if b.NComp() != a.NComp() {
		t.Fail()
	}
	if b.Host()[1][1] != 42 {
		t.Fail()
	}
}
