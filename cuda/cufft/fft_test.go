package cufft

import (
	"fmt"
	"github.com/mumax/3/cuda/cu"
	"unsafe"
)

func ExampleFFT1D() {
	N := 8

	hostIn := make([]float32, N)
	hostIn[0] = 1

	devIn := cu.MemAlloc(int64(len(hostIn)) * cu.SIZEOF_FLOAT32)
	defer cu.MemFree(devIn)
	cu.MemcpyHtoD(devIn, unsafe.Pointer(&hostIn[0]), devIn.Bytes())

	hostOut := make([]complex64, N/2+1)
	devOut := cu.MemAlloc(int64(len(hostOut)) * cu.SIZEOF_COMPLEX64)
	defer cu.MemFree(devOut)

	plan := Plan1d(N, R2C, 1)
	defer plan.Destroy()
	plan.ExecR2C(devIn, devOut)

	cu.MemcpyDtoH(unsafe.Pointer(&hostOut[0]), devOut, devOut.Bytes())

	fmt.Println("hostIn:", hostIn)
	fmt.Println("hostOut:", hostOut)

	// Output:
	// hostIn: [1 0 0 0 0 0 0 0]
	// hostOut: [(1+0i) (1+0i) (1+0i) (1-0i) (1+0i)]
}
