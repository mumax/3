package nc

import (
	"testing"
)

func BenchmarkGpuBufferRecycle(b *testing.B) {
	LOG = false
	b.StopTimer()

	// init and warmup
	InitSize(1, 1, 1)
	A, B, C := GpuBuffer(), GpuBuffer(), GpuBuffer()
	RecycleGpu(A, B, C)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		A := GpuBuffer()
		RecycleGpu(A)
	}
}

func BenchmarkBufferRecycle(b *testing.B) {
	LOG = false
	b.StopTimer()

	// init and warmup
	InitSize(1, 1, 1)
	A, B, C := Buffer(), Buffer(), Buffer()
	Recycle(A, B, C)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		A := Buffer()
		Recycle(A)
	}
}
