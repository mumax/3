package nc

type VectorBlock [3]Block

func MakeVectorBlock(N0, N1, N2 int) VectorBlock {
	checkPositive(N0, N1, N2)
	var sliced [VECCOMP]Block
	for i := range sliced {
		sliced[i] = make(Block, N0)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float32, N1)
		}
	}
	list := make([]float32, VECCOMP*N0*N1*N2)
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*N0+j)*N1+k)*N2+0 : ((i*N0+j)*N1+k)*N2+N2]
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

func (v VectorBlock) N0() int {
	return len(v[0])
}

func (v VectorBlock) N1() int {
	return len(v[0][0])
}

func (v VectorBlock) N2() int {
	return len(v[0][0][0])
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

// Get the i'th Vector element.
//func (v VectorBlock) Get(i int) Vector {
//	return Vector{v[X][i], v[Y][i], v[Z][i]}
//}
//
//// Set the i'th Vector element.
//func (v VectorSlice) Set(i int, value Vector) {
//	v[X][i] = value[X]
//	v[Y][i] = value[Y]
//	v[Z][i] = value[Z]
//}
