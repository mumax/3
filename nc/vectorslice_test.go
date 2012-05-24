package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorSlice() {
	N := 10
	vec := MakeVectorSlice(N)
	fmt.Println("vec.N():", vec.N(), "len(vec):", len(vec)) //←[ inlining call to VectorSlice.N  ExampleVectorSlice ... argument does not escape]

	// Take the Y-component.
	y := vec[Y]
	fmt.Println("y.N():", y.N(), "len(y):", len(y)) //←[ inlining call to Slice.N  ExampleVectorSlice ... argument does not escape]

	y.Memset(2)
	vec.Set(7, Vector{4, 5, 6})                        //←[ inlining call to VectorSlice.Set]
	fmt.Println("y:", y)                               //←[ ExampleVectorSlice ... argument does not escape]
	fmt.Println("vec:", vec)                           //←[ ExampleVectorSlice ... argument does not escape]
	fmt.Println("vec.Get(7):", vec.Get(7))             //←[ inlining call to VectorSlice.Get  ExampleVectorSlice ... argument does not escape]
	fmt.Println("vec.Contiguous():", vec.Contiguous()) //←[ ExampleVectorSlice ... argument does not escape]

	// Output: 
	// vec.N(): 10 len(vec): 3
	// y.N(): 10 len(y): 10
	// y: [2 2 2 2 2 2 2 5 2 2]
	// vec: [[0 0 0 0 0 0 0 4 0 0] [2 2 2 2 2 2 2 5 2 2] [0 0 0 0 0 0 0 6 0 0]]
	// vec.Get(7): [4 5 6]
	// vec.Contiguous(): [0 0 0 0 0 0 0 4 0 0 2 2 2 2 2 2 2 5 2 2 0 0 0 0 0 0 0 6 0 0]

}

func BenchmarkVectorSliceSet(bench *testing.B) { //←[ BenchmarkVectorSliceSet bench does not escape]
	N := 100
	vec := MakeVectorSlice(N)
	v := Vector{1, 2, 3}
	for i := 0; i < bench.N; i++ {
		vec.Set(42, v) //←[ inlining call to VectorSlice.Set]
	}
}

func BenchmarkVectorSliceSetInline(bench *testing.B) { //←[ BenchmarkVectorSliceSetInline bench does not escape]
	N := 100
	vec := MakeVectorSlice(N)
	for i := 0; i < bench.N; i++ {
		vec[X][42] = 1
		vec[Y][42] = 2
		vec[Z][42] = 3
	}
}

func BenchmarkVectorSliceGet(bench *testing.B) { //←[ BenchmarkVectorSliceGet bench does not escape]
	N := 100
	vec := MakeVectorSlice(N)
	var v Vector
	for i := 0; i < bench.N; i++ {
		v = vec.Get(42) //←[ inlining call to VectorSlice.Get]
	}
	use(v) //←[ inlining call to use]
}

func BenchmarkVectorSliceContiguous(bench *testing.B) { //←[ BenchmarkVectorSliceContiguous bench does not escape]
	N := 100
	vec := MakeVectorSlice(N)
	var s Slice
	for i := 0; i < bench.N; i++ {
		s = vec.Contiguous()
	}
	use(s) //←[ inlining call to use]
}
