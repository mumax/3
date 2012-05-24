package nc

import (
	"fmt"
	"testing"
)

func ExampleSlice() {
	N := 10
	vec := MakeVectorSlice(N)
	fmt.Println("vec.N():", vec.N(), "len(vec):", len(vec))

	// Take the Y-component.
	y := vec.VectorComp(Y)
	fmt.Println("y.N():", y.N(), "len(y):", len(y))

	y.Set(2)
	fmt.Println("y:", y)
	fmt.Println("vec:", vec)

	// Output: 
	// vec.N(): 10 len(vec): 30
	// y.N(): 10 len(y): 10
	// y: [2 2 2 2 2 2 2 2 2 2]
	//vec: [0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0]
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

func BenchmarkNop(bench *testing.B) {
	for i := 0; i < bench.N; i++ {
	}
}
