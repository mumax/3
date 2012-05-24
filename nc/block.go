package nc

import ()

// Block is a [][][]float32 with square layout and contiguous underlying storage:
// 	len(block[0]) == len(block[1]) == ...
// 	len(block[i][0]) == len(block[i][1]) == ...
// 	len(block[i][j][0]) == len(block[i][j][1]) == ...
type Block [][][]float32

func MakeBlock(N0, N1, N2 int) Block {
	checkPositive(N0, N1, N2)
	sliced := make([][][]float32, N0)
	for i := range sliced {
		sliced[i] = make([][]float32, N1)
	}
	storage := make([]float32, N0*N1*N2)
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = storage[(i*N1+j)*N2+0 : (i*N1+j)*N2+N2]
		}
	}
	return Block(sliced)
}

// Total number of scalar elements.
func (v Block) N() int {
	return len(v) * len(v[0]) * len(v[0][0])
}

func (v Block) N0() int {
	return len(v)
}

func (v Block) N1() int {
	return len(v[0])
}

func (v Block) N2() int {
	return len(v[0][0])
}

// Returns the contiguous underlying storage.
// Contains first all X component, than Y, than Z.
func (v Block) Contiguous() Slice {
	return [][][]float32(v)[0][0][:v.N()]
}

// Set all elements to a.
func (v Block) Memset(a float32) {
	storage := v.Contiguous()
	for i := range storage {
		storage[i] = a
	}
}

func checkPositive(N0, N1, N2 int) {
	if N0 < 1 {
		Panic("MakeBlock: N0 out of range")
	}
	if N1 < 1 {
		Panic("MakeBlock: N1 out of range")
	}
	if N2 < 1 {
		Panic("MakeBlock: N2 out of range")
	}
}
