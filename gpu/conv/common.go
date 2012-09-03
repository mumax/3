package conv

// common code for all convolutions.

import "github.com/barnex/cuda4/cu"

// CUDA root stream.
const stream0 cu.Stream = 0

// Output size of R2C FFT with given logic size, expressed in floats.
func fftR2COutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], 2 * (logicSize[2]/2 + 1)}
}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
