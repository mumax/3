package cufft

import (
	"github.com/barnex/cuda4/cu"
	"testing"
	"unsafe"
)

func ExampleFFT1D(test *testing.T) {
	N := 5

	hostIn := make([]float32, N)
	devIn := cu.MemAlloc(int64(len(hostIn)) * cu.SIZEOF_FLOAT32)
	defer cu.MemFree(&devIn)
	cu.MemcpyHtoD(devIn, unsafe.Pointer(&hostIn[0]), devIn.Bytes())

	hostOut := make([]complex64, N/2+1)
	devOut := cu.MemAlloc(int64(len(hostOut)) * cu.SIZEOF_COMPLEX64)
	defer cu.MemFree(&devOut)

	plan := Plan1d(N, R2C, 1)
	plan.ExecR2C(devIn, devOut)

	cu.MemcpyDtoH(unsafe.Pointer(&hostOut[0]), devOut)

	fmt.Println(hostOut)

	// Output:
	// a
}
