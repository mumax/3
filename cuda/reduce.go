package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	k_reducesum(in.DevPtr(0), out, 0, in.Len(), reducecfg)
	return copyback(out)
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	out := reduceBuf(0)
	for c := 0; c < nComp; c++ {
		k_reducedot(a.DevPtr(c), b.DevPtr(c), out, 0, a.Len(), reducecfg) // all components add to out
	}
	return copyback(out)
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	k_reducemaxabs(in.DevPtr(0), out, 0, in.Len(), reducecfg)
	return copyback(out)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	out := reduceBuf(0)
	k_reducemaxvecnorm2(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

//// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0)
	k_reducemaxvecdiff2(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

var reduceBuffers chan unsafe.Pointer // pool of 1-float CUDA buffers for reduce

// return a 1-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	str := stream()
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), math.Float32bits(initVal), 1, str)
	syncAndRecycle(str)
	return buf
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result float32
	cu.MemcpyDtoH(unsafe.Pointer(&result), cu.DevicePtr(uintptr(buf)), cu.SIZEOF_FLOAT32)
	reduceBuffers <- buf
	return result
}

// initialize pool of 1-float CUDA reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan unsafe.Pointer, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(1 * cu.SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
// 8 is typ. number of multiprocessors.
// could be improved but takes hardly ~1% of execution time
var reducecfg = &config{Grid: cu.Dim3{8, 1, 1}, Block: cu.Dim3{REDUCE_BLOCKSIZE, 1, 1}}
