package nc

import (
	"fmt"
	"testing"
)

func ExampleVectorString() {
	v := Vector{1, 2, 3}
	fmt.Println(v)
	// Output: [1 2 3]
}

func TestVectorAdd(test *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{4, 5, 6}
	sum := Vector{5, 7, 9}
	if a.Add(b) != sum {
		test.Fail()
	}
	if a != (Vector{1, 2, 3}) || b != (Vector{4, 5, 6}) {
		test.Fail()
	}
}

func TestVectorSub(test *testing.T) {
	a := Vector{1, 2, 6}
	b := Vector{4, -5, 3}
	sub := Vector{-3, 7, 3}
	if a.Sub(b) != sub {
		test.Error(a.Sub(b))
	}
	if a != (Vector{1, 2, 6}) || b != (Vector{4, -5, 3}) {
		test.Fail()
	}
}

func TestVectorCross(test *testing.T) {
	a := Vector{1, 0, 0}
	b := Vector{0, 1, 0}
	cross := Vector{0, 0, 1}
	if a.Cross(b) != cross {
		test.Error(a.Cross(b))
	}
	if a != (Vector{1, 0, 0}) || b != (Vector{0, 1, 0}) {
		test.Fail()
	}
}

func BenchmarkVectorAdd(bench *testing.B) {
	var a, b Vector
	for i := 0; i < bench.N; i++ {
		a.Add(b)
	}
}
