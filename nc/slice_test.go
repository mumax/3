package nc

import (
	"testing"
)

func BenchmarkNop(bench *testing.B) { //←[ BenchmarkNop bench does not escape]
	for i := 0; i < bench.N; i++ {
	}
}
