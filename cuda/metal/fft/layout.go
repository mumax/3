// Package fft contains the Apple Metal FFT bridge and small, deterministic
// reference routines used to validate cuFFT-compatible layout semantics.
package fft

import "fmt"

// Layout describes a contiguous row-major transform. The final dimension is
// the fastest-moving dimension, matching cufftPlan{1d,2d,3d}.
type Layout struct {
	Dimensions []int
	Batch      int
}

// NewLayout validates and copies a transform shape.
func NewLayout(dimensions []int, batch int) (Layout, error) {
	if len(dimensions) == 0 || len(dimensions) > 3 {
		return Layout{}, fmt.Errorf("metal fft: rank must be between 1 and 3, got %d", len(dimensions))
	}
	if batch < 1 {
		return Layout{}, fmt.Errorf("metal fft: batch must be positive, got %d", batch)
	}
	dims := append([]int(nil), dimensions...)
	count := batch
	maxCount := int(^uint(0)>>1) / 8
	for axis, size := range dims {
		if size < 1 {
			return Layout{}, fmt.Errorf("metal fft: dimension %d must be positive, got %d", axis, size)
		}
		if count > maxCount/size {
			return Layout{}, fmt.Errorf("metal fft: shape %v with batch %d exceeds addressable storage", dimensions, batch)
		}
		count *= size
	}
	return Layout{Dimensions: dims, Batch: batch}, nil
}

// RealShape returns the MPSGraph tensor shape for real data. A leading batch
// dimension is included only when Batch is greater than one; transform axes
// always refer to the trailing dimensions.
func (l Layout) RealShape() []int {
	shape := append([]int(nil), l.Dimensions...)
	if l.Batch > 1 {
		shape = append([]int{l.Batch}, shape...)
	}
	return shape
}

// HermitianShape returns the interleaved-complex tensor shape produced by a
// real-to-complex transform. cuFFT and MPSGraph both retain N/2+1 values along
// the fastest-moving (last) axis.
func (l Layout) HermitianShape() []int {
	shape := l.RealShape()
	shape[len(shape)-1] = shape[len(shape)-1]/2 + 1
	return shape
}

// TransformAxes returns the trailing tensor axes transformed by the plan.
func (l Layout) TransformAxes() []int {
	offset := 0
	if l.Batch > 1 {
		offset = 1
	}
	axes := make([]int, len(l.Dimensions))
	for i := range axes {
		axes[i] = offset + i
	}
	return axes
}

// RealCount is the number of float32 values in the real tensor.
func (l Layout) RealCount() int {
	return product(l.RealShape())
}

// HermitianCount is the number of complex64 values in the packed tensor.
func (l Layout) HermitianCount() int {
	return product(l.HermitianShape())
}

// LastDimensionIsOdd tells MPSGraph how to reconstruct the final dimension
// during a Hermitian-to-real transform.
func (l Layout) LastDimensionIsOdd() bool {
	return l.Dimensions[len(l.Dimensions)-1]%2 != 0
}

func product(values []int) int {
	result := 1
	for _, value := range values {
		result *= value
	}
	return result
}
