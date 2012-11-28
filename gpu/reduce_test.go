package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"math"
	"testing"
)

func init() { core.LOG = false }

func TestReduceSum(t *testing.T) {
	LockCudaThread()
	N := 1000
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = float32(i)
	}
	str := cu.StreamCreate()
	result := reduceSum(input.Device(), str)
	if result != 499500 {
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

func TestReduceMaxDiff(t *testing.T) {
	LockCudaThread()
	N := 100001
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = -float32(i) / 100
	}
	input2 := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in2 := input2.Host()
	for i := range in2 {
		in2[i] = float32(i) / 100
	}
	str := cu.StreamCreate()
	result := reduceMaxDiff(input.Device(), input2.Device(), str)
	if result != 2000 {
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


func TestReduceMaxVecNorm(t *testing.T) {
	LockCudaThread()
	N := 1234
	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
	in := input.Host()
	for i := range in {
		in[i] = -float32(i) / 1000
	}
	str := cu.StreamCreate()
	x:= input.Device()
	result := reduceMaxNorm(x, x, x, str)
	want := math.Sqrt(3) * 1233. / 1000.
	if math.Abs(result - want)>1e-7 {
		t.Error("got:", result, "want:", want)
	}
}

