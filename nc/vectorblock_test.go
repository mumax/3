package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorBlock() {
	N0, N1, N2 := 1, 2, 3
	block := MakeVectorBlock(N0, N1, N2)
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
	b := MakeBlock(N0, N1, N2)
	if b.N0() != N0 {
		test.Fail()
	}
	if b.N1() != N1 {
		test.Fail()
	}
	if b.N2() != N2 {
		test.Fail()
	}
	if b.N() != N0*N1*N2 {
		test.Fail()
	}
	if b.N() != len(b.Contiguous()) {
		test.Fail()
	}
}
