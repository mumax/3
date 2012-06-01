package cu

import (
	"testing"
	"unsafe"
	//"fmt"
)

func TestModule(test *testing.T) {
	mod := ModuleLoad("/testdata/testmodule.ptx")
	f := mod.GetFunction("testMemset")

	N := 1000
	N4 := 4 * int64(N)
	a := make([]float32, N)
	A := MemAlloc(N4)
	defer A.Free()
	aptr := unsafe.Pointer(&a[0])
	MemcpyHtoD(A, aptr, N4)

	var array uintptr
	array = uintptr(A)

	var value float32
	value = 42

	var n int
	n = N / 2

	args := []unsafe.Pointer{unsafe.Pointer(&array), unsafe.Pointer(&value), unsafe.Pointer(&n)}
	block := 128
	grid := DivUp(N, block)
	shmem := 0
	LaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, 0, args)

	MemcpyDtoH(aptr, A, N4)
	for i := 0; i < N/2; i++ {
		if a[i] != 42 {
			test.Fail()
		}
	}
	for i := N / 2; i < N; i++ {
		if a[i] != 0 {
			test.Fail()
		}
	}
	//fmt.Println(a)
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
