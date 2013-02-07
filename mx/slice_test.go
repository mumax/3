package mx

import "testing"

func TestSlice(t *testing.T) {
	LockCudaThread()
	N := 100
	a := MakeSlice(3, N)
	defer a.Free()
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

	//b := a.Comp(1)

}

func TestSliceFree(t *testing.T) {
	LockCudaThread()
	length := 128 * 1024 * 1024
	N := 17
	// not freeing would attempt to allocate 17GB.
	for i := 0; i < N; i++ {
		a := MakeSlice(2, length)
		a.Free()
	}
	a := MakeSlice(2, length)
	a.Free()
	a.Free() // test double-free
}
