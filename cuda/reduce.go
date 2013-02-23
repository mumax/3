package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	kernel.K_reducesum(in.DevPtr(0), out, 0, in.Len(), gr, bl)
	return copyback(out)
}

// Maximum of all elements.
//func Max(in *data.Slice) float32 {
//	return reduce1(in, -math.MaxFloat32, kernel.K_reducemax)
//}

// Minimum of all elements.
//func Min(in *data.Slice) float32 {
//	return reduce1(in, math.MaxFloat32, kernel.K_reducemin)
//}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	kernel.K_reducemaxabs(in.DevPtr(0), out, 0, in.Len(), gr, bl)
	return copyback(out)
}

//// Maximum difference between the two arrays.
//// 	max_i abs(a[i] - b[i])
//func MaxDiff(a, b safe.Float32s) float32 {
//	return reduce2(a, b, 0, ptx.K_reducemaxdiff)
//}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	out := reduceBuf(0)
	kernel.K_reducemaxvecnorm2(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), gr, bl)
	return math.Sqrt(float64(copyback(out)))
}

//// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0)
	kernel.K_reducemaxvecdiff2(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), gr, bl)
	return math.Sqrt(float64(copyback(out)))
}

//
//// Vector dot product.
//func Dot(x1, x2 safe.Float32s) float32 {
//	return reduce2(x1, x2, 0, ptx.K_reducedot)
//}
//
//
//// general reduce wrapper for 2 input arrays
//func reduce2(in1, in2 safe.Float32s, init float32, f func(in1, in2, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
//	Argument(in1.Len() == in2.Len())
//	out := reduceBuf(init)
//	defer reduceRecycle(out)
//	gr, bl := reduceConf()
//	f(in1.Pointer(), in2.Pointer(), out.Pointer(), init, in1.Len(), gr, bl)
//	return copyback(out)
//}
//
//// general reduce wrapper for 3 input arrays
//func reduce3(in1, in2, in3 safe.Float32s, init float32, f func(in1, in2, in3, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
//	N := in1.Len()
//	Argument(in2.Len() == N && in3.Len() == N)
//	out := reduceBuf(init)
//	defer reduceRecycle(out)
//	gr, bl := reduceConf()
//	f(in1.Pointer(), in2.Pointer(), in3.Pointer(), out.Pointer(), init, N, gr, bl)
//	return copyback(out)
//}
//

var reduceBuffers chan unsafe.Pointer // pool of 1-float CUDA buffers for reduce

// return a 1-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	str := kernel.Stream()
	cu.MemsetD32Async(cu.DevicePtr(buf), math.Float32bits(initVal), 1, str)
	kernel.SyncAndRecycle(str)
	return buf
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result_ [1]float32
	result := result_[:]
	cu.MemcpyDtoH(unsafe.Pointer(&result[0]), cu.DevicePtr(buf), 1*cu.SIZEOF_FLOAT32)
	reduceBuffers <- buf
	return result_[0]
}

// initialize pool of 1-float CUDA reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan unsafe.Pointer, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- memAlloc(1 * cu.SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
var (
	bl = cu.Dim3{kernel.REDUCE_BLOCKSIZE, 1, 1}
	gr = cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.
	// could be improved but takes hardly ~1% of execution time
)
