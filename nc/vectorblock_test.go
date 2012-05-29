package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorBlock() {
	N0, N1, N2 := 1, 2, 3
	size := [3]int{N0, N1, N2}
	block := MakeVectorBlock(size)
	fmt.Println("block.NVector():", block.NVector()) // N0*N1*N2
	fmt.Println("block.NFloat():", block.NFloat())   // 3*N0*N1*N2

	storage := block.Contiguous()
	for i := range storage[X] {
		storage[X][i] = float32(i)
		storage[Y][i] = float32(10 * i)
		storage[Z][i] = float32(100 * i)
	}

	fmt.Println("block:", block)
	fmt.Println("block.Contiguous():", storage)
	fmt.Println("block[X]:", block[X])
	fmt.Println("block[X].Contiguous():", block[X].Contiguous())

	// Output:
	// block.NVector(): 6
	// block.NFloat(): 18
	// block: [[[[0 1 2] [3 4 5]]] [[[0 10 20] [30 40 50]]] [[[0 100 200] [300 400 500]]]]
	// block.Contiguous(): [[0 1 2 3 4 5] [0 10 20 30 40 50] [0 100 200 300 400 500]]
	// block[X]: [[[0 1 2] [3 4 5]]]
	// block[X].Contiguous(): [0 1 2 3 4 5]
}

func TestVectorBlock(test *testing.T) {
	N0, N1, N2 := 2, 3, 4
	size := [3]int{N0, N1, N2}
	b := MakeVectorBlock(size)
	if b.BlockSize() != size {
		test.Fail()
	}
	if b.NVector() != N0*N1*N2 {
		test.Fail()
	}
}

//TODO: move to vectorslice
func BenchmarkVectorBlockNormalize(bench *testing.B) {
	bench.StopTimer()
	N0, N1, N2 := 16, 1024, 1024
	size := [3]int{N0, N1, N2}
	b := MakeVectorBlock(size)
	b.Memset(7)
	bytes := 4 * int64(b.NFloat())
	bench.SetBytes(bytes)
	bench.Log("size: ", size, "=", bytes/(1024*1024), "MB")
	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		b.Normalize()
	}
}
