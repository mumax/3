package mx

import "testing"

func TestSlice(t *testing.T) {
	LockCudaThread()
	N := 100
	a := MakeSlice(3, N)
	Log(a)
}
