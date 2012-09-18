package conv

import (
	"flag"
	"fmt"
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/mag"
	"testing"
)

// some test sizes
var (
	N0s = []int{1}
	N1s = []int{2, 3, 4, 8, 16, 32, 48, 63, 64, 65}
	N2s = []int{2, 3, 4, 8, 16, 32, 48, 64, 128, 255, 256, 257, 1024}
)

func init() { flag.Parse() }

func TestBasic(test *testing.T) {
	gpu.LockCudaThread()
	*gpu.Flag_pagelock = false

	core.LOG = false
	for _, N0 := range N0s {
		for _, N1 := range N1s {
			for _, N2 := range N2s {
				testConvSize(test, NewBasic, N0, N1, N2)
			}
		}
	}
}

func TestSymmetric(test *testing.T) {
	gpu.LockCudaThread()
	*gpu.Flag_pagelock = false

	core.LOG = false
	for _, N0 := range N0s {
		for _, N1 := range N1s {
			for _, N2 := range N2s {
				testConvSize(test, NewSymmetric, N0, N1, N2)
			}
		}
	}
}

func testConvSize(test *testing.T, f Constructor, N0, N1, N2 int) {
	defer func() {
		err := recover()
		if err != nil {
			test.Error(N0, N1, N2, err)
		} else {
			fmt.Println(N0, N1, N2, "OK")
		}
	}()
	C := 1e-9
	mesh := core.NewMesh(N0, N1, N2, C, 2*C, 3*C)
	acc := 2
	kern := mag.BruteKernel(mesh.ZeroPadded(), acc)

	c := f(mesh.GridSize(), kern)
	c.Input()[0][N0/2][0][0] = 1
	c.Input()[1][0][N1/2][0] = 2
	c.Input()[2][0][0][N2/2] = 3
	c.Exec()
}

