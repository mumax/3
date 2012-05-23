package nc

import (
	"testing"
)

func BenchmarkString(bench *testing.B) {
	v := Vector{1, 2, 3}
	for i := 0; i < bench.N; i++ {
		v.String()
	}
}


func BenchmarkAdd(bench *testing.B) {
	a := Vector{1, 2, 3}
	b := Vector{1, 2, 3}
	for i := 0; i < bench.N; i++ {
		a.Add(b)
	}
}
