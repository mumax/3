package nc

import ()

// Block is a [][][]float32 with square layout and contiguous underlying storage:
// 	len(block[0]) == len(block[1]) == ...
// 	len(block[i][0]) == len(block[i][1]) == ...
// 	len(block[i][j][0]) == len(block[i][j][1]) == ...
type Block [][][]float32

func MakeBlock(N [3]int) Block {
	checkSize(N[:])
	sliced := make([][][]float32, N[0])
	for i := range sliced {
		sliced[i] = make([][]float32, N[1])
	}
	storage := make([]float32, N[0]*N[1]*N[2])
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = storage[(i*N[1]+j)*N[2]+0 : (i*N[1]+j)*N[2]+N[2]]
		}
	}
	return Block(sliced)
}

// Total number of scalar elements.
func (v Block) NFloat() int {
	return len(v) * len(v[0]) * len(v[0][0])
}

func (v Block) BlockSize() [3]int {
	return [3]int{len(v), len(v[0]), len(v[0][0])}
}

// Returns the contiguous underlying storage.
// Contains first all X component, than Y, than Z.
func (v Block) Contiguous() Slice {
	return [][][]float32(v)[0][0][:v.NFloat()]
}

// Set all elements to a.
func (v Block) Memset(a float32) {
	storage := v.Contiguous()
	for i := range storage {
		storage[i] = a
	}
}

func checkSize(size []int) {
	for i, s := range size {
		if s < 1 {
			Panic("MakeBlock: N", i, " out of range:", s)
		}
	}
}
