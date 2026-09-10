//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

// Philox-derived portions: Copyright 2010-2012, D. E. Shaw Research.
// BSD-3-Clause; see the Random123 notice in the repository LICENSE.

package rng

import (
	"math"
	"testing"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
)

func TestMetalPhiloxRandom123KnownAnswer(t *testing.T) {
	requireMetal(t)
	generator, err := CreateGenerator(100)
	if err != nil {
		t.Fatal(err)
	}
	if err := SetSeed(generator, 0); err != nil {
		t.Fatal(err)
	}
	buffer := gpuAlloc(t, 16)
	defer gpuFree(t, buffer)
	if err := GenerateRaw(generator, uintptr(buffer), 1); err != nil {
		t.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	got := [4]uint32{}
	if err := metal.CopyToHost(unsafe.Pointer(&got[0]), buffer, 16); err != nil {
		t.Fatal(err)
	}
	want := [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}
	if got != want {
		t.Fatalf("Metal Philox4x32-10 zero vector = %08x, want %08x", got, want)
	}
}

func TestMetalNormalDeterminismAndStatistics(t *testing.T) {
	requireMetal(t)
	const n = 1 << 20
	const seed = uint64(0x0123456789abcdef)
	const mean = float32(1.25)
	const standardDeviation = float32(2.5)
	generator, err := CreateGenerator(100)
	if err != nil {
		t.Fatal(err)
	}
	buffer := gpuAlloc(t, n*4)
	defer gpuFree(t, buffer)

	generate := func() []float32 {
		t.Helper()
		if err := SetSeed(generator, seed); err != nil {
			t.Fatal(err)
		}
		if err := GenerateNormal(generator, uintptr(buffer), n, mean, standardDeviation); err != nil {
			t.Fatal(err)
		}
		if err := metal.Sync(); err != nil {
			t.Fatal(err)
		}
		samples := make([]float32, n)
		if err := metal.CopyToHost(unsafe.Pointer(&samples[0]), buffer, n*4); err != nil {
			t.Fatal(err)
		}
		return samples
	}
	first := generate()
	second := generate()

	var sum, sumSquares float64
	minimum := math.Inf(1)
	maximum := math.Inf(-1)
	for i, sample := range first {
		if sample != second[i] {
			t.Fatalf("same seed is not deterministic at %d: %g != %g", i, sample, second[i])
		}
		value := float64(sample)
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("sample %d is not finite: %g", i, value)
		}
		minimum = math.Min(minimum, value)
		maximum = math.Max(maximum, value)
		sum += value
		sumSquares += value * value
	}
	gotMean := sum / n
	gotVariance := sumSquares/float64(n) - gotMean*gotMean
	wantVariance := float64(standardDeviation * standardDeviation)
	if math.Abs(gotMean-float64(mean)) > 0.02 {
		t.Fatalf("mean=%g, want approximately %g", gotMean, mean)
	}
	if math.Abs(gotVariance-wantVariance) > 0.08 {
		t.Fatalf("variance=%g, want approximately %g", gotVariance, wantVariance)
	}
	if minimum >= float64(mean-3*standardDeviation) ||
		maximum <= float64(mean+3*standardDeviation) {
		t.Fatalf("sample range [%g,%g] does not cover both 3-sigma tails", minimum, maximum)
	}
}

func BenchmarkMetalGenerateNormal1M(b *testing.B) {
	if err := metal.Initialize(); err != nil {
		b.Fatal(err)
	}
	const n = 1 << 20
	generator, err := CreateGenerator(100)
	if err != nil {
		b.Fatal(err)
	}
	buffer, err := metal.Alloc(n * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(buffer); err != nil {
			b.Error(err)
		}
	}()
	if err := GenerateNormal(generator, uintptr(buffer), n, 0, 1); err != nil {
		b.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(n * 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := GenerateNormal(generator, uintptr(buffer), n, 0, 1); err != nil {
			b.Fatal(err)
		}
		if err := metal.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func requireMetal(t *testing.T) {
	t.Helper()
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
}

func gpuAlloc(t *testing.T, bytes int64) unsafe.Pointer {
	t.Helper()
	pointer, err := metal.Alloc(bytes)
	if err != nil {
		t.Fatal(err)
	}
	return pointer
}

func gpuFree(t *testing.T, pointer unsafe.Pointer) {
	t.Helper()
	if err := metal.Free(pointer); err != nil {
		t.Error(err)
	}
}
