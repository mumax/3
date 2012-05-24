package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorString() {
	v := Vector{1, 2, 3}
	fmt.Println(v) //←[ ExampleVectorString ... argument does not escape]
	// Output: [1 2 3]
}

func TestVectorAdd(test *testing.T) { //←[ TestVectorAdd test does not escape]
	a := Vector{1, 2, 3}
	b := Vector{4, 5, 6}
	sum := Vector{5, 7, 9}
	if a.Add(b) != sum { //←[ inlining call to Vector.Add]
		test.Fail() //←[ inlining call to Fail  TestVectorAdd test.common does not escape]
	}
	if a != (Vector{1, 2, 3}) || b != (Vector{4, 5, 6}) {
		test.Fail() //←[ inlining call to Fail  TestVectorAdd test.common does not escape]
	}
}

func TestVectorSub(test *testing.T) { //←[ TestVectorSub test does not escape]
	a := Vector{1, 2, 6}
	b := Vector{4, -5, 3}
	sub := Vector{-3, 7, 3}
	if a.Sub(b) != sub { //←[ inlining call to Vector.Sub]
		test.Error(a.Sub(b)) //←[ inlining call to Vector.Sub  TestVectorSub test.common does not escape  TestVectorSub ... argument does not escape]
	}
	if a != (Vector{1, 2, 6}) || b != (Vector{4, -5, 3}) {
		test.Fail() //←[ inlining call to Fail  TestVectorSub test.common does not escape]
	}
}

func BenchmarkVectorAdd(bench *testing.B) { //←[ BenchmarkVectorAdd bench does not escape]
	var a, b Vector
	for i := 0; i < bench.N; i++ {
		a.Add(b) //←[ inlining call to Vector.Add]
	}
}
