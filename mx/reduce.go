package mx

import (
	"code.google.com/p/mx3/ptx"
	"code.google.com/p/mx3/streams"
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// general reduce wrapper for one input array
func reduce1(in *Slice, init float32, f func(in, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
	out := reduceBuf(init)
	defer reduceRecycle(out)
	gr, bl := reduceConf()
	Argument(in.NComp() == 1)
	f(in.DevPtr(0), out, init, in.Len(), gr, bl)
	return copyback(out)
}

// Sum of all elements.
func Sum(in *Slice) float32 {
	return reduce1(in, 0, ptx.K_reducesum)
}

//
//// Maximum of all elements.
//func Max(in safe.Float32s) float32 {
//	return reduce1(in, -math.MaxFloat32, ptx.K_reducemax)
//}
//
//// Minimum of all elements.
//func Min(in safe.Float32s) float32 {
//	return reduce1(in, math.MaxFloat32, ptx.K_reducemin)
//}
//
//// Maximum of absolute values of all elements.
//func MaxAbs(in safe.Float32s) float32 {
//	return reduce1(in, 0, ptx.K_reducemaxabs)
//}
//
//// Maximum difference between the two arrays.
//// 	max_i abs(a[i] - b[i])
//func MaxDiff(a, b safe.Float32s) float32 {
//	return reduce2(a, b, 0, ptx.K_reducemaxdiff)
//}
//
//// Maximum of the norms of all vectors (x[i], y[i], z[i]).
//// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
//func MaxVecNorm(x, y, z safe.Float32s) float64 {
//	r := reduce3(x, y, z, 0, ptx.K_reducemaxvecnorm2)
//	return math.Sqrt(float64(r))
//}
//
//// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
//func MaxVecDiff(x1, y1, z1, x2, y2, z2 safe.Float32s) float64 {
//	r := reduce6(x1, y1, z1, x2, y2, z2, 0, ptx.K_reducemaxvecdiff2)
//	return math.Sqrt(float64(r))
//}
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
//// general reduce wrapper for 6 input arrays
//func reduce6(in1, in2, in3, in4, in5, in6 safe.Float32s, init float32, f func(in1, in2, in3, in4, in5, in6, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
//	N := in1.Len()
//	Argument(in2.Len() == N && in3.Len() == N && in4.Len() == N && in5.Len() == N && in6.Len() == N)
//	out := reduceBuf(init)
//	defer reduceRecycle(out)
//	gr, bl := reduceConf()
//	f(in1.Pointer(), in2.Pointer(), in3.Pointer(), in4.Pointer(), in5.Pointer(), in6.Pointer(), out.Pointer(), init, N, gr, bl)
//	return copyback(out)
//}

var reduceBuffers chan cu.DevicePtr // pool of 1-float CUDA buffers for reduce

// recycle a 1-float CUDA reduction buffer
func reduceRecycle(buf cu.DevicePtr) {
	reduceBuffers <- buf
}

// return a 1-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) cu.DevicePtr {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	str := streams.Get()
	cu.MemsetD32Async(buf, math.Float32bits(initVal), 1, str)
	streams.SyncAndRecycle(str)
	return buf
}

func copyback(buf cu.DevicePtr) float32 {
	var result_ [1]float32
	result := result_[:]
	cu.MemcpyDtoH(unsafe.Pointer(&result[0]), buf, 1*cu.SIZEOF_FLOAT32)
	return result_[0]
}

// initialize pool of 1-float CUDA reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan cu.DevicePtr, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(1 * cu.SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
func reduceConf() (gridDim, blockDim cu.Dim3) {
	blockDim = cu.Dim3{512, 1, 1}
	gridDim = cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.
	// could be improved but takes hardly ~1% of execution time
	return
}
