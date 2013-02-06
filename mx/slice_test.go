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

}
