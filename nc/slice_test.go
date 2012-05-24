package nc

import (
	"testing"
)

func BenchmarkNop(bench *testing.B) {
	for i := 0; i < bench.N; i++ {
	}
}
