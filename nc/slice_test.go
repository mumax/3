package nc

import (
	"testing"
)

func benchmarknop(bench *testing.B) {
	for i := 0; i < bench.N; i++ {
	}
}

func BenchmarkCopy(bench *testing.B) {
	bench.StopTimer()
	N := 3 * 16 * 1024 * 1024
	s := MakeSlice(N)
	t := MakeSlice(N)
	bench.SetBytes(4 * int64(N))
	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		copy(s, t)
	}
}

func BenchmarkSliceCopy(bench *testing.B) {
	bench.StopTimer()
	N := 3 * 16 * 1024 * 1024
	s := MakeSlice(N)
	d := MakeSlice(N)
	bench.SetBytes(4 * int64(N))
	bench.StartTimer()
	for I := 0; I < bench.N; I++ {
		for i, v := range s {
			d[i] = v
		}
	}
}
