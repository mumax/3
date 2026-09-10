package fft

import (
	"fmt"
	"math"
)

// ReferenceR2C computes a small, unnormalised row-major real-to-Hermitian DFT.
// It is intentionally O(N²): production uses MPSGraph on the GPU, while this
// routine is a dependency-free correctness oracle for layout and sign tests.
func ReferenceR2C(input []float32, dimensions []int) ([]complex64, error) {
	layout, err := NewLayout(dimensions, 1)
	if err != nil {
		return nil, err
	}
	if len(input) != layout.RealCount() {
		return nil, fmt.Errorf("metal fft reference R2C: got %d values, want %d", len(input), layout.RealCount())
	}

	outputShape := layout.HermitianShape()
	output := make([]complex64, product(outputShape))
	inputCoord := make([]int, len(dimensions))
	outputCoord := make([]int, len(dimensions))

	for outputIndex := range output {
		unflatten(outputIndex, outputShape, outputCoord)
		var realPart, imaginaryPart float64
		for inputIndex, value := range input {
			unflatten(inputIndex, dimensions, inputCoord)
			phase := phaseDot(inputCoord, outputCoord, dimensions)
			angle := -2 * math.Pi * phase
			realPart += float64(value) * math.Cos(angle)
			imaginaryPart += float64(value) * math.Sin(angle)
		}
		output[outputIndex] = complex(float32(realPart), float32(imaginaryPart))
	}
	return output, nil
}

// ReferenceC2R computes the unnormalised inverse of a packed Hermitian tensor.
// As with cuFFT, applying C2R after R2C yields product(dimensions)*input.
func ReferenceC2R(input []complex64, dimensions []int) ([]float32, error) {
	layout, err := NewLayout(dimensions, 1)
	if err != nil {
		return nil, err
	}
	halfShape := layout.HermitianShape()
	if len(input) != product(halfShape) {
		return nil, fmt.Errorf("metal fft reference C2R: got %d values, want %d", len(input), product(halfShape))
	}

	output := make([]float32, layout.RealCount())
	outputCoord := make([]int, len(dimensions))
	frequencyCoord := make([]int, len(dimensions))

	for outputIndex := range output {
		unflatten(outputIndex, dimensions, outputCoord)
		var realPart float64
		for frequencyIndex := 0; frequencyIndex < layout.RealCount(); frequencyIndex++ {
			unflatten(frequencyIndex, dimensions, frequencyCoord)
			value := hermitianAt(input, halfShape, dimensions, frequencyCoord)
			phase := phaseDot(outputCoord, frequencyCoord, dimensions)
			angle := 2 * math.Pi * phase
			realPart += float64(real(value))*math.Cos(angle) - float64(imag(value))*math.Sin(angle)
		}
		output[outputIndex] = float32(realPart)
	}
	return output, nil
}

func hermitianAt(half []complex64, halfShape, fullShape, frequency []int) complex64 {
	last := len(fullShape) - 1
	if frequency[last] < halfShape[last] {
		return half[flatten(frequency, halfShape)]
	}

	mirror := make([]int, len(frequency))
	for axis, value := range frequency {
		if value == 0 {
			mirror[axis] = 0
		} else {
			mirror[axis] = fullShape[axis] - value
		}
	}
	value := half[flatten(mirror, halfShape)]
	return complex(real(value), -imag(value))
}

func phaseDot(lhs, rhs, dimensions []int) float64 {
	var result float64
	for axis := range dimensions {
		result += float64(lhs[axis]*rhs[axis]) / float64(dimensions[axis])
	}
	return result
}

func flatten(coord, dimensions []int) int {
	index := 0
	for axis, size := range dimensions {
		index = index*size + coord[axis]
	}
	return index
}

func unflatten(index int, dimensions, coord []int) {
	for axis := len(dimensions) - 1; axis >= 0; axis-- {
		coord[axis] = index % dimensions[axis]
		index /= dimensions[axis]
	}
}
