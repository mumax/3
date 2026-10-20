//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package fft

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
)

func TestMetalFFT3DRoundTripEvenAndOdd(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	for _, dimensions := range [][]int{{2, 3, 4}, {2, 3, 5}} {
		t.Run(shapeName(dimensions), func(t *testing.T) {
			layout, err := NewLayout(dimensions, 1)
			if err != nil {
				t.Fatal(err)
			}
			input := make([]float32, layout.RealCount())
			for i := range input {
				input[i] = float32(math.Sin(float64(i)*0.29) + 0.25*math.Cos(float64(i)*0.17))
			}

			realBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
			spectrumBuffer := mustMetalAlloc(t, int64(layout.HermitianCount()*8))
			roundTripBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
			defer mustMetalFree(t, realBuffer)
			defer mustMetalFree(t, spectrumBuffer)
			defer mustMetalFree(t, roundTripBuffer)
			if err := metal.CopyToDevice(realBuffer, unsafe.Pointer(&input[0]), int64(len(input)*4)); err != nil {
				t.Fatal(err)
			}

			forward, err := CreatePlan(layout, RealToComplex)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := DestroyPlan(forward); err != nil {
					t.Error(err)
				}
			}()
			inverse, err := CreatePlan(layout, ComplexToReal)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := DestroyPlan(inverse); err != nil {
					t.Error(err)
				}
			}()

			if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
				t.Fatal(err)
			}
			if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(roundTripBuffer), 1); err != nil {
				t.Fatal(err)
			}
			if err := metal.Sync(); err != nil {
				t.Fatal(err)
			}

			output := make([]float32, len(input))
			if err := metal.CopyToHost(unsafe.Pointer(&output[0]), roundTripBuffer, int64(len(output)*4)); err != nil {
				t.Fatal(err)
			}
			scale := float32(layout.RealCount())
			for i := range input {
				want := scale * input[i]
				tolerance := 8e-4 * math.Max(1, math.Abs(float64(want)))
				if difference := math.Abs(float64(output[i] - want)); difference > tolerance {
					t.Fatalf("dims=%v output[%d]=%g, want %g ± %g", dimensions, i, output[i], want, tolerance)
				}
			}
		})
	}
}

func TestMetalFFTR2CImpulsePacking(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	layout, err := NewLayout([]int{3, 4, 5}, 1)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, layout.RealCount())
	input[0] = 1
	spectrum := make([]complex64, layout.HermitianCount())

	realBuffer := mustMetalAlloc(t, int64(len(input)*4))
	spectrumBuffer := mustMetalAlloc(t, int64(len(spectrum)*8))
	defer mustMetalFree(t, realBuffer)
	defer mustMetalFree(t, spectrumBuffer)
	if err := metal.CopyToDevice(realBuffer, unsafe.Pointer(&input[0]), int64(len(input)*4)); err != nil {
		t.Fatal(err)
	}
	plan, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(plan); err != nil {
			t.Error(err)
		}
	}()
	if err := Execute(plan, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		t.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := metal.CopyToHost(unsafe.Pointer(&spectrum[0]), spectrumBuffer, int64(len(spectrum)*8)); err != nil {
		t.Fatal(err)
	}
	for i, value := range spectrum {
		if difference := cmplxAbs(value - 1); difference > 2e-5 {
			t.Fatalf("packed spectrum[%d]=%v, want 1+0i", i, value)
		}
	}
}

func TestMetalFFTR2CMatchesReferenceSpectrum(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	layout, err := NewLayout([]int{2, 3, 5}, 1)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, layout.RealCount())
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.41) + 0.07*float64(i))
	}
	want, err := ReferenceR2C(input, layout.Dimensions)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]complex64, layout.HermitianCount())

	realBuffer := mustMetalAlloc(t, int64(len(input)*4))
	spectrumBuffer := mustMetalAlloc(t, int64(len(got)*8))
	defer mustMetalFree(t, realBuffer)
	defer mustMetalFree(t, spectrumBuffer)
	if err := metal.CopyToDevice(realBuffer, unsafe.Pointer(&input[0]), int64(len(input)*4)); err != nil {
		t.Fatal(err)
	}
	plan, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(plan); err != nil {
			t.Error(err)
		}
	}()
	if err := Execute(plan, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		t.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := metal.CopyToHost(unsafe.Pointer(&got[0]), spectrumBuffer, int64(len(got)*8)); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		tolerance := 5e-4 * math.Max(1, cmplxAbs(want[i]))
		if difference := cmplxAbs(got[i] - want[i]); difference > tolerance {
			t.Fatalf("packed spectrum[%d]=%v, want %v ± %g", i, got[i], want[i], tolerance)
		}
	}
}

func BenchmarkMetalFFT3DRoundTrip(b *testing.B) {
	if err := metal.Initialize(); err != nil {
		b.Fatal(err)
	}
	layout, err := NewLayout([]int{32, 128, 128}, 1)
	if err != nil {
		b.Fatal(err)
	}
	realBuffer, err := metal.Alloc(int64(layout.RealCount() * 4))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(realBuffer); err != nil {
			b.Error(err)
		}
	}()
	spectrumBuffer, err := metal.Alloc(int64(layout.HermitianCount() * 8))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(spectrumBuffer); err != nil {
			b.Error(err)
		}
	}()
	outputBuffer, err := metal.Alloc(int64(layout.RealCount() * 4))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(outputBuffer); err != nil {
			b.Error(err)
		}
	}()
	forward, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(forward); err != nil {
			b.Error(err)
		}
	}()
	inverse, err := CreatePlan(layout, ComplexToReal)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(inverse); err != nil {
			b.Error(err)
		}
	}()

	if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		b.Fatal(err)
	}
	if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(outputBuffer), 1); err != nil {
		b.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(layout.RealCount() * 4 * 2))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
			b.Fatal(err)
		}
		if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(outputBuffer), 1); err != nil {
			b.Fatal(err)
		}
		if err := metal.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func mustMetalAlloc(t *testing.T, bytes int64) unsafe.Pointer {
	t.Helper()
	pointer, err := metal.Alloc(bytes)
	if err != nil {
		t.Fatal(err)
	}
	return pointer
}

func mustMetalFree(t *testing.T, pointer unsafe.Pointer) {
	t.Helper()
	if err := metal.Free(pointer); err != nil {
		t.Error(err)
	}
}

func shapeName(dimensions []int) string {
	result := ""
	for i, dimension := range dimensions {
		if i != 0 {
			result += "x"
		}
		result += fmt.Sprint(dimension)
	}
	return result
}
