package nc

import (
	"math"
	"testing"
)

func TestSqrtf(test *testing.T) { //←[ TestSqrtf test does not escape]
	f := Sqrtf(2)
	if f != 1.41421356237 {
		test.Error("Sqrtf(2):", f) //←[ TestSqrtf test.common does not escape  TestSqrtf ... argument does not escape]
	}
	f = Sqrtf(-1)
	if !math.IsNaN(float64(f)) { //←[ inlining call to math.IsNaN]
		test.Error("Sqrtf(-1):", f) //←[ TestSqrtf test.common does not escape  TestSqrtf ... argument does not escape]
	}
}

func BenchmarkSqrtf(bench *testing.B) { //←[ BenchmarkSqrtf bench does not escape]
	var a float32
	for i := 0; i < bench.N; i++ {
		a = Sqrtf(15.7)
	}
	use(a) //←[ inlining call to use]
}

// Portable sqrtf.
func sqrtf(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// Avoids "unused variable" for benchmark funcs.
func use(x interface{}) {} //←[ can inline use  use x does not escape]

func BenchmarkSqrtfPortable(bench *testing.B) { //←[ BenchmarkSqrtfPortable bench does not escape]
	var a float32
	for i := 0; i < bench.N; i++ {
		a = sqrtf(15.7)
	}
	use(a) //←[ inlining call to use]
}
