package mx

//import (
//	"math"
//	"testing"
//)
//
//func TestReduceSum(t *testing.T) {
//	LockCudaThread()
//	N := 1000
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = float32(i)
//	}
//	result := Sum(input.Device())
//	if result != 499500 {
//		t.Error("got:", result)
//	}
//}
//
//func TestReduceMax(t *testing.T) {
//	LockCudaThread()
//	N := 100000
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = float32(i) / 100
//	}
//	result := Max(input.Device())
//	if result != 999.99 {
//		t.Error("got:", result)
//	}
//}
//
//func TestReduceMaxAbs(t *testing.T) {
//	LockCudaThread()
//	N := 100000
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = -float32(i) / 100
//	}
//	result := MaxAbs(input.Device())
//	if result != 999.99 {
//		t.Error("got:", result)
//	}
//}
//
//func TestReduceMaxDiff(t *testing.T) {
//	LockCudaThread()
//	N := 100001
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = -float32(i) / 100
//	}
//	input2 := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in2 := input2.Host()
//	for i := range in2 {
//		in2[i] = float32(i) / 100
//	}
//	result := MaxDiff(input.Device(), input2.Device())
//	if result != 2000 {
//		t.Error("got:", result)
//	}
//}
//
//func TestReduceMin(t *testing.T) {
//	LockCudaThread()
//	N := 100
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = float32(i) + 100
//	}
//	result := Min(input.Device())
//	if result != 100 {
//		t.Error("got:", result)
//	}
//}
//
//func BenchmarkReduceSum(b *testing.B) {
//	core.LOG = false
//	b.StopTimer()
//	LockCudaThread()
//	const N = 32 * 1024 * 1024
//	input := nimble.MakeSlice(N, nimble.GPUMemory)
//	b.SetBytes(N * 4)
//	b.StartTimer()
//	for i := 0; i < b.N; i++ {
//		Sum(input.Device())
//	}
//}
//
//func TestReduceMaxVecNorm(t *testing.T) {
//	LockCudaThread()
//	N := 1234
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = -float32(i) / 1000
//	}
//	x := input.Device()
//	result := MaxVecNorm(x, x, x)
//	want := math.Sqrt(3) * 1233. / 1000.
//	if math.Abs(result-want) > 1e-7 {
//		t.Error("got:", result, "want:", want)
//	}
//}
//
//func TestReduceMaxVecDiff(t *testing.T) {
//	LockCudaThread()
//	N := 1234
//	input := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in := input.Host()
//	for i := range in {
//		in[i] = -float32(i) / 1000
//	}
//	x := input.Device()
//	input2 := nimble.MakeSlice(N, nimble.UnifiedMemory)
//	in2 := input2.Host()
//	for i := range in2 {
//		in2[i] = 0
//	}
//	y := input2.Device()
//	result := MaxVecDiff(x, x, x, y, y, y)
//	want := math.Sqrt(3) * 1233. / 1000.
//	if math.Abs(result-want) > 1e-7 {
//		t.Error("got:", result, "want:", want)
//	}
//}
