package gpu

import (
	"code.google.com/p/nimble-cube/core"
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
		in[i] = float32(i) / 100
	}
	str := cu.StreamCreate()
	result := reduceSum(input.Device(), str)
	if result != 499950 {
		t.Error("got:", result)
	}
}

func TestReduceMax(t *testing.T) {
	LockCudaThread()
	N := 100000
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = float32(i) / 100
	}
	str := cu.StreamCreate()
	result := reduceMax(input.Device(), str)
	if result != 999.99 {
		t.Error("got:", result)
	}
}

func TestReduceMaxAbs(t *testing.T) {
	LockCudaThread()
	N := 100000
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = -float32(i) / 100
	}
	str := cu.StreamCreate()
	result := reduceMaxAbs(input.Device(), str)
	if result != 999.99 {
		t.Error("got:", result)
	}
}

func TestReduceMin(t *testing.T) {
	LockCudaThread()
	N := 10033
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = float32(i) - 100
	}
	str := cu.StreamCreate()
	result := reduceMin(input.Device(), str)
	if result != -100 {
		t.Error("got:", result)
	}
}

func BenchmarkReduceSum(b *testing.B) {
	core.LOG = false
	b.StopTimer()
	LockCudaThread()
	const N = 32 * 1024 * 1024
	input := nimble.MakeSlice(N, nimble.GPUMemory)
	str := cu.StreamCreate()
	b.SetBytes(N * 4)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		reduceSum(input.Device(), str)
	}
}
