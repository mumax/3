package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorSlice() {
	N := 10
	vec := MakeVectorSlice(N)
	fmt.Println("vec.NVector():", vec.NVector())
	fmt.Println("vec.NFloat):", vec.NFloat())

	// Take the Y-component.
	y := vec[Y]
	fmt.Println("y.N():", y.N())

	y.Memset(2)
	vec.Set(7, Vector{4, 5, 6})
	fmt.Println("y:", y)
	fmt.Println("vec:", vec)
	fmt.Println("vec.Get(7):", vec.Get(7))
	fmt.Println("vec.Contiguous():", vec.Contiguous())

	// Output: 
	// vec.NVector(): 10
	// vec.NFloat): 30
	// y.N(): 10
	// y: [2 2 2 2 2 2 2 5 2 2]
	// vec: [[0 0 0 0 0 0 0 4 0 0] [2 2 2 2 2 2 2 5 2 2] [0 0 0 0 0 0 0 6 0 0]]
	// vec.Get(7): [4 5 6]
	// vec.Contiguous(): [0 0 0 0 0 0 0 4 0 0 2 2 2 2 2 2 2 5 2 2 0 0 0 0 0 0 0 6 0 0]
}

func BenchmarkVectorSliceSet(bench *testing.B) {
	N := 100
	vec := MakeVectorSlice(N)
	v := Vector{1, 2, 3}
	for i := 0; i < bench.N; i++ {
		vec.Set(42, v)
	}
}

func BenchmarkVectorSliceSetInline(bench *testing.B) {
	N := 100
	vec := MakeVectorSlice(N)
	for i := 0; i < bench.N; i++ {
		vec[X][42] = 1
		vec[Y][42] = 2
		vec[Z][42] = 3
	}
}

func BenchmarkVectorSliceGet(bench *testing.B) {
	N := 100
	vec := MakeVectorSlice(N)
	var v Vector
	for i := 0; i < bench.N; i++ {
		v = vec.Get(42)
	}
	use(v)
}

func BenchmarkVectorSliceContiguous(bench *testing.B) {
	N := 100
	vec := MakeVectorSlice(N)
	var s Slice
	for i := 0; i < bench.N; i++ {
		s = vec.Contiguous()
	}
	use(s)
}
