package mx

import "testing"

func BenchmarkAlloc(b *testing.B) {
	b.StopTimer()
	arr := make([][]float32, b.N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		arr[i] = make([]float32, 1)
	}
}
