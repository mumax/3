package conv

// common code for all convolutions.

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"github.com/barnex/fmath"
	"nimble-cube/core"
)

// CUDA root stream.
const stream0 cu.Stream = 0

// Output size of R2C FFT with given logic size, expressed in floats.
func fftR2COutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], 2 * (logicSize[2]/2 + 1)}
}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Extract real parts, copy them from src to dst.
// In the meanwhile, check if imaginary parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
func scaleRealParts(dstList []float32, src safe.Float32s, scale float32) {
	core.Assert(2*len(dstList) == src.Len())
	srcList := src.Host()

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maximg := float32(0.)
	maxreal := float32(0.)
	for i := 0; i < src.Len()/2; i++ {
		dstList[i] = srcList[2*i] * scale
		if fmath.Abs(srcList[2*i+0]) > maxreal {
			maxreal = fmath.Abs(srcList[2*i+0])
		}
		if fmath.Abs(srcList[2*i+1]) > maximg {
			maximg = fmath.Abs(srcList[2*i+1])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	//core.Debug("FFT Kernel max imaginary part=", maximg)
	//core.Debug("FFT Kernel max real part=", maxreal)
	core.Debug("FFT Kernel max imaginary/real part=", maximg/maxreal)
	if maximg/maxreal > 1e-5 { // TODO: is this reasonable?
		core.Panicf("xc: FFT Kernel max imaginary/real part=%v", maximg/maxreal)
	}
}
