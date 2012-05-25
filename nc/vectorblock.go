package nc

import (
	//"math"
)

type VectorBlock [3]Block

func MakeVectorBlock(N [3]int) VectorBlock {
	checkSize(N[:])
	var sliced [VECCOMP]Block
	for i := range sliced {
		sliced[i] = make(Block, N[0])
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float32, N[1])
		}
	}
	list := make([]float32, VECCOMP*N[0]*N[1]*N[2])
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*N[0]+j)*N[1]+k)*N[2]+0 : ((i*N[0]+j)*N[1]+k)*N[2]+N[2]]
			}
		}
	}
	return VectorBlock(sliced)
}

// Total number of scalar elements.
func (v VectorBlock) NFloat() int {
	return len(v) * len(v[0]) * len(v[0][0]) * len(v[0][0][0])
}

// Total number of vector elements.
func (v VectorBlock) NVector() int {
	return len(v[0]) * len(v[0][0]) * len(v[0][0][0])
}

func (v VectorBlock) BlockSize() [3]int {
	return [3]int{len(v[0]), len(v[0][0]), len(v[0][0][0])}
}

// Returns the contiguous underlying storage.
// Contains first all X component, than Y, than Z.
func (v VectorBlock) Contiguous() Slice {
	return (([3]Block)(v))[0][0][0][:v.NFloat()]
}

// Set all elements to a.
func (v VectorBlock) Memset(a float32) {
	storage := v.Contiguous()
	for i := range storage {
		storage[i] = a
	}
}

func (v VectorBlock) Normalize() {
	sx := v[X].Contiguous()
	sy := v[Y].Contiguous()
	sz := v[Z].Contiguous()
	for i,x := range sx {
		y := sy[i]
		z := sz[i]
		norm:=float32(7)
		//norm := 1 / float32(math.Sqrt(float64(x*x+y*y+z*z)))
		sx[i] = x * norm
		sy[i] = y * norm
		sz[i] = z * norm
	}
}
