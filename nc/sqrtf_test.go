package nc

import (
	"math"
	"testing"
)

func TestSqrtf(test *testing.T) {
	f := Sqrtf(2)
	if f != 1.41421356237 {
		test.Error("Sqrtf(2):", f)
	}
	f = Sqrtf(-1)
	if !math.IsNaN(float64(f)) {
		test.Error("Sqrtf(-1):", f)
	}
}

func BenchmarkSqrtf(bench *testing.B) {
	var a float32
	for i := 0; i < bench.N; i++ {
		a = Sqrtf(15.7)
	}
	use(a)
}

// Portable sqrtf.
func sqrtf(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// Avoids "unused variable" for benchmark funcs.
func use(x interface{}) {}

func BenchmarkSqrtfPortable(bench *testing.B) {
	var a float32
	for i := 0; i < bench.N; i++ {
		a = sqrtf(15.7)
	}
	use(a)
}
