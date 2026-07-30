package rng

import (
	"math"
	"testing"
)

func TestPhilox4x32Random123KnownAnswer(t *testing.T) {
	got := Philox4x32([4]uint32{}, [2]uint32{})
	want := [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}
	if got != want {
		t.Fatalf("Philox4x32-10 zero vector = %08x, want %08x", got, want)
	}
}

func TestReferenceNormalDeterministicAndCounterSeparated(t *testing.T) {
	a := make([]float32, 17)
	b := make([]float32, 17)
	c := make([]float32, 17)
	ReferenceNormal(a, 1234, 99, 0, 1)
	ReferenceNormal(b, 1234, 99, 0, 1)
	ReferenceNormal(c, 1234, 104, 0, 1)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("determinism failed at %d: %g != %g", i, a[i], b[i])
		}
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) {
			t.Fatalf("sample %d is not finite: %g", i, a[i])
		}
	}
	equal := true
	for i := range a {
		equal = equal && a[i] == c[i]
	}
	if equal {
		t.Fatal("different non-overlapping counter ranges produced the same samples")
	}
}

func TestUniformMappingStaysOpenAfterFloat32Rounding(t *testing.T) {
	minimum := float32(uniformOpen(0))
	maximum := float32(uniformOpen(^uint32(0)))
	if !(minimum > 0 && minimum < 1) {
		t.Fatalf("minimum uniform value %g is outside (0,1)", minimum)
	}
	if !(maximum > 0 && maximum < 1) {
		t.Fatalf("maximum uniform value %g is outside (0,1)", maximum)
	}
	if minimum != float32(math.Ldexp(1, -24)) ||
		maximum != float32(1-math.Ldexp(1, -24)) {
		t.Fatalf("uniform endpoints are [%g,%g], want [2^-24,1-2^-24]", minimum, maximum)
	}
}

func TestReferenceNormalStatistics(t *testing.T) {
	const n = 1 << 18
	const mean = float32(1.25)
	const stddev = float32(2.5)
	samples := make([]float32, n)
	ReferenceNormal(samples, 0x0123456789abcdef, 0, mean, stddev)

	var sum, sumSquares float64
	minimum := math.Inf(1)
	maximum := math.Inf(-1)
	for _, sample := range samples {
		value := float64(sample)
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("normal sample is not finite: %g", value)
		}
		minimum = math.Min(minimum, value)
		maximum = math.Max(maximum, value)
		sum += value
		sumSquares += value * value
	}
	gotMean := sum / n
	gotVariance := sumSquares/float64(n) - gotMean*gotMean
	wantVariance := float64(stddev * stddev)
	if math.Abs(gotMean-float64(mean)) > 0.02 {
		t.Fatalf("mean=%g, want approximately %g", gotMean, mean)
	}
	if math.Abs(gotVariance-wantVariance) > 0.08 {
		t.Fatalf("variance=%g, want approximately %g", gotVariance, wantVariance)
	}
	if minimum >= float64(mean-3*stddev) || maximum <= float64(mean+3*stddev) {
		t.Fatalf("sample range [%g,%g] does not cover both 3-sigma tails", minimum, maximum)
	}
}
