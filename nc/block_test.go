package nc

import (
	"fmt"
	"testing"
)

func ExampleBlock() {
	N0, N1, N2 := 2, 3, 4
	block := MakeBlock(N0, N1, N2)
	fmt.Println("block.N():", block.N()) // N0*N1*N2

	storage := block.Contiguous()
	for i := range storage{
		storage[i] = float32(i)
	}
	
	fmt.Println("storage:", storage)
	fmt.Println("block:", block)

	// Output: 
	// block.N(): 24
	// storage: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
	// block: [[[0 1 2 3] [4 5 6 7] [8 9 10 11]] [[12 13 14 15] [16 17 18 19] [20 21 22 23]]]

}

func BenchmarkBlockContiguous(bench *testing.B) {
	bench.StopTimer()
	N0, N1, N2 := 20, 300, 400
	b := MakeBlock(N0, N1, N2)
	var s Slice
	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		s = b.Contiguous()
	}
	use(s)
}

func BenchmarkBlockN(bench *testing.B) {
	bench.StopTimer()
	N0, N1, N2 := 20, 300, 400
	b := MakeBlock(N0, N1, N2)
	bench.StartTimer()
	n:=0
	for i := 0; i < bench.N; i++ {
		n = b.N()
	}
	use(n)
}
