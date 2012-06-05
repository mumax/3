package nc

import (
	"fmt"
	"testing"
)

func ExampleBlock() {
	N0, N1, N2 := 2, 3, 4
	size := [3]int{N0, N1, N2}
	block := MakeBlock(size)
	fmt.Println("block.NFloat():", block.NFloat()) // N0*N1*N2

	storage := block.Contiguous()
	for i := range storage {
		storage[i] = float32(i)
	}

	fmt.Println("storage:", storage)
	fmt.Println("block:", block)

	// Output: 
	// block.NFloat(): 24
	// storage: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
	// block: [[[0 1 2 3] [4 5 6 7] [8 9 10 11]] [[12 13 14 15] [16 17 18 19] [20 21 22 23]]]
}

func TestBlock(test *testing.T) {
	N0, N1, N2 := 2, 3, 4
	size := [3]int{N0, N1, N2}
	b := MakeBlock(size)
	if b.BlockSize() != size {
		test.Fail()
	}
	if b.NFloat() != N0*N1*N2 {
		test.Fail()
	}
	if b.NFloat() != len(b.Contiguous()) {
		test.Fail()
	}
}

func BenchmarkBlockContiguous(bench *testing.B) {
	bench.StopTimer()
	N0, N1, N2 := 20, 300, 400
	size := [3]int{N0, N1, N2}
	b := MakeBlock(size)
	var s []float32
	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		s = b.Contiguous()
	}
	use(s)
}

func BenchmarkBlockNFloat(bench *testing.B) {
	bench.StopTimer()
	N0, N1, N2 := 20, 300, 400
	size := [3]int{N0, N1, N2}
	b := MakeBlock(size)
	bench.StartTimer()
	n := 0
	for i := 0; i < bench.N; i++ {
		n = b.NFloat()
	}
	use(n)
}
