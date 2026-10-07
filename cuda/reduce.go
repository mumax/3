package cuda

import (
	"math"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

//#include "reduce.h"
import "C"

// Block size for reduce kernels.
const REDUCE_BLOCKSIZE = C.REDUCE_BLOCKSIZE

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	partials := reduceBufN(reducecfg.nBlocks(), 0)
	k_reducesum_async(in.DevPtr(0), partials, 0, in.Len(), reducecfg)
	host := copybackSlice(partials, reducecfg.nBlocks())

	// Add elements of partials on CPU
	var result float32
	for _, v := range host {
		result += v
	}
	return result
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	partials := reduceBufN(reducecfg.nBlocks(), 0)
	// not async over components
	for c := 0; c < nComp; c++ {
		k_reducedot_async(a.DevPtr(c), b.DevPtr(c), partials, 0, a.Len(), reducecfg) // all components add to out
	}
	host := copybackSlice(partials, reducecfg.nBlocks())

	// Add elements of partials on CPU
	var result float32
	for _, v := range host {
		result += v
	}
	return result
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	k_reducemaxabs_async(in.DevPtr(0), out, 0, in.Len(), reducecfg)
	return copyback(out)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
//
//	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	out := reduceBuf(0)
	k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0)
	k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

var reduceBuffers chan unsafe.Pointer                // pool of 1-float CUDA buffers for reduce
var reduceBuffersN = map[int64]chan unsafe.Pointer{} // pool of N-float CUDA buffers for reduce

// return a 1-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), math.Float32bits(initVal), 1, stream0)
	return buf
}

// return an N-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBufN(N int64, initVal float32) unsafe.Pointer {
	const Nbufs = 128
	q, ok := reduceBuffersN[N]
	if !ok { // Create buffer channel for this N if it doesn't exist
		q = make(chan unsafe.Pointer, Nbufs)
		for i := 0; i < Nbufs; i++ {
			q <- MemAlloc(N * cu.SIZEOF_FLOAT32)
		}
		reduceBuffersN[N] = q
	}

	buf := <-q
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), math.Float32bits(initVal), N, stream0)
	return buf
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result float32
	MemCpyDtoH(unsafe.Pointer(&result), buf, cu.SIZEOF_FLOAT32)
	reduceBuffers <- buf
	return result
}

// copy back a slice of length N from GPU
func copybackSlice(buf unsafe.Pointer, N int64) []float32 {
	result := make([]float32, N)
	MemCpyDtoH(unsafe.Pointer(&result[0]), buf, N*cu.SIZEOF_FLOAT32)
	reduceBuffersN[N] <- buf
	return result
}

// initialize pool of 1-float CUDA reduction buffers
func initReduceBuf() {
	const Nbufs = 128
	reduceBuffers = make(chan unsafe.Pointer, Nbufs)
	for i := 0; i < Nbufs; i++ {
		reduceBuffers <- MemAlloc(1 * cu.SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
// Grid size is set during cuda.Init to equal the number of multiprocessors.
var reducecfg *config
