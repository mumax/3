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
	for i := range storage {
		storage[i] = float32(i)
	}

	fmt.Println("block:", block)
	fmt.Println("block.Contiguous():", storage)
	fmt.Println("block[X]:", block[X])
	fmt.Println("block[X].Contiguous():", block[X].Contiguous())

	// Output:
	// block.NVector(): 6
	// block.NFloat(): 18
	// block: [[[[0 1 2] [3 4 5]]] [[[6 7 8] [9 10 11]]] [[[12 13 14] [15 16 17]]]]
	// block.Contiguous(): [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17]
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
	if b.NFloat() != len(b.Contiguous()) {
		test.Fail()
	}
}
