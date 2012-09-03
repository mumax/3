package conv

import (
	"fmt"
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/mag"
	"testing"
)

// some test sizes
var (
	N0s = []int{1}
	N1s = []int{2, 3, 8, 32, 48, 64, 63}
	N2s = []int{2, 3, 8, 32, 48, 64, 128, 255}
)

func TestGeneral(test *testing.T) {
	gpu.LockCudaThread()

	core.LOG = false
	for _, N0 := range N0s {
		for _, N1 := range N1s {
			for _, N2 := range N2s {
				testGeneralSize(test, N0, N1, N2)
			}
		}
	}
}

func testGeneralSize(test *testing.T, N0, N1, N2 int) {
	defer func() {
		err := recover()
		if err != nil {
			test.Error(N0, N1, N2, err)
		} else {
			fmt.Println(N0, N1, N2, "OK")
		}
	}()
	size := [3]int{N0, N1, N2}
	core.Log("size:", size)

	ksize := core.PadSize(size, [3]int{0, 0, 0})
	acc := 2
	kern := mag.BruteKernel(ksize, [3]float64{1, 2, 3}, [3]int{0, 0, 0}, acc)

	c := NewGeneral(size, kern)
	c.Input()[0][N0/2][0][0] = 1
	c.Input()[1][0][N1/2][0] = 2
	c.Input()[2][0][0][N2/2] = 3
	c.Exec()
}

///func TestSymmetric(test *testing.T) {
///	//core.LOG = false
///	for _, N0 := range N0s {
///		for _, N1 := range N1s {
///			for _, N2 := range N2s {
///				size := [3]int{N0, N1, N2}
///				core.Log("size:", size)
///				//cellsize := [3]float64{1e-9, 1e-9, 1e-9}
///				N := core.Prod(size)
///
///				input := core.MakeVectors(size)
///				input[0][N0/2][N1/2][N2/2] = 1
///				input[1][N0/2][N1/2][N2/2] = 2
///				input[2][N0/2][N1/2][N2/2] = 3
///				output := core.MakeVectors(size)
///
///				ksize := core.PadSize(size, [3]int{0, 0, 0})
///				var kern [3][3][][][]float32
///				for i := 0; i < 3; i++ {
///					for j := i; j < 3; j++ {
///						kern[i][j] = core.MakeFloats(ksize)
///					}
///				}
///				kern[0][0][0][0][0] = 1
///				kern[1][1][0][0][0] = 2
///				kern[2][2][0][0][0] = 3
///
///				conv := NewSymmetric(input, output, kern)
///				conv.Push(N)
///				conv.Pull(N)
///
///				//core.Print(output)
///
///				if output[0][N0/2][N1/2][N2/2] != 1 ||
///					output[1][N0/2][N1/2][N2/2] != 4 ||
///					output[2][N0/2][N1/2][N2/2] != 9 {
///					test.Error("size=", size, "got:", output[0][N0/2][N1/2][N2/2], output[1][N0/2][N1/2][N2/2], output[2][N0/2][N1/2][N2/2])
///					core.Log("FAIL size:", size)
///				} else {
///					core.Log(" OK  size:", size)
///				}
///				//conv.Free()
///			}
///		}
///	}
///
///}

//func BenchmarkConv1_2DSmall(b *testing.B) {
//	b.StopTimer()
//
//	size := [3]int{1, 128, 128}
//	core.InitSize(size[0], size[1], size[2])
//	core.InitCellSize(1e-9, 1e-9, 1e-9)
//	N := prod(size)
//
//	in := make([]float32, 3*N)
//	input := [3][]float32{in[0*N : 1*N], in[1*N : 2*N], in[2*N : 3*N]}
//
//	out := make([]float32, 3*N)
//	output := [3][]float32{out[0*N : 1*N], out[1*N : 2*N], out[2*N : 3*N]}
//
//	conv := NewConv1(input, output, size)
//
//	b.SetBytes(int64(prod(size)) * 4 * 2) // *2: xfer back and forth
//
//	conv.Test()
//
//	core.DEBUG = false
//	core.LOG = false
//
//	// warmup
//	conv.Push(core.N())
//	conv.Pull(core.N())
//	b.StartTimer()
//	for i := 0; i < b.N; i++ {
//		conv.Push(core.N())
//		conv.Pull(core.N())
//	}
//	b.StopTimer()
//	core.Cleanup()
//}
