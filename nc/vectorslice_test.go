package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorSlice() {
	N := 10
	vec := MakeVectorSlice(N)
	fmt.Println("vec.N():", vec.N(), "len(vec):", len(vec))

	// Take the Y-component.
	y := vec.VectorComp(Y)
	fmt.Println("y.N():", y.N(), "len(y):", len(y))

	y.Memset(2)
	vec[X][7] = 4
	vec[Y][7] = 5
	vec[Z][7] = 6
	fmt.Println("y:", y)
	fmt.Println("vec:", vec)
	//fmt.Println("vec.Get(7):", vec.Get(7)) 

	// Output: 
	// vec.N(): 10 len(vec): 3
	// y.N(): 10 len(y): 10
	// y: [2 2 2 2 2 2 2 5 2 2]
	// vec: [[0 0 0 0 0 0 0 4 0 0] [2 2 2 2 2 2 2 5 2 2] [0 0 0 0 0 0 0 6 0 0]]

}

func BenchmarkVectorSliceComponent(bench *testing.B) {
	N := 100
	vec := MakeVectorSlice(N)
	var a Slice
	for i := 0; i < bench.N; i++ {
		a = vec.VectorComp(Y)
	}
	use(a)
}

//func BenchmarkVectorSliceSet(bench *testing.B) {
//	N := 100
//	vec := MakeVectorSlice(N)
//	for i := 0; i < bench.N; i++ {
//		vec.Set(42, Vector{1, 2, 3}) 
//	}
//}
