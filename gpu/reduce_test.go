package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"testing"
)

func TestReduceSum(t *testing.T) {
	LockCudaThread()
	N := 10000
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = float32(i)/100
	}
	str := cu.StreamCreate()
	result := reduce_sum(input.Device(), str)
	if result != 499950 {
		t.Error("got:", result)
	}
}

func BenchmarkReduceSum(b *testing.B) {
	b.StopTimer()
	LockCudaThread()
	const N = 1024*1024
	input := nimble.MakeSlice(N, nimble.GPUMemory)
	str := cu.StreamCreate()
	b.SetBytes(N*4)
	b.StartTimer()
	for i:=0; i<b.N; i++{
		reduce_sum(input.Device(), str)
	}
}
