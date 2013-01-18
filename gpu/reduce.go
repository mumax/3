package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"math"
)

// Sum of all elements.
func Sum(in safe.Float32s, stream cu.Stream) float32 {
	return reduce1(in, 0, ptx.K_reducesum)
}

// Maximum of all elements.
func Max(in safe.Float32s, stream cu.Stream) float32 {
	return reduce1(in, -math.MaxFloat32, ptx.K_reducemax)
}

// Minimum of all elements.
func Min(in safe.Float32s, stream cu.Stream) float32 {
	return reduce1(in, math.MaxFloat32, ptx.K_reducemin)
}

// Maximum of absolute values of all elements.
func MaxAbs(in safe.Float32s, stream cu.Stream) float32 {
	return reduce1(in, 0, ptx.K_reducemaxabs)
}

// general reduce wrapper for one input array
func reduce1(in safe.Float32s, init float32, f func(in, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
	out := reduceBuf(init)
	defer reduceRecycle(out)
	gr, bl := reduceConf()
	f(in.Pointer(), out.Pointer(), init, in.Len(), gr, bl)
	return copyback(out)
}

// general reduce wrapper for 2 input arrays
func reduce2(in1, in2 safe.Float32s, init float32, f func(in1, in2, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
	core.Assert(in1.Len() == in2.Len())
	out := reduceBuf(init)
	defer reduceRecycle(out)
	gr, bl := reduceConf()
	f(in1.Pointer(), in2.Pointer(), out.Pointer(), init, in1.Len(), gr, bl)
	return copyback(out)
}

// general reduce wrapper for 3 input arrays
func reduce3(in1, in2, in3 safe.Float32s, init float32, f func(in1, in2, in3, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
	N := in1.Len()
	core.Assert(in2.Len() == N && in3.Len() == N)
	out := reduceBuf(init)
	defer reduceRecycle(out)
	gr, bl := reduceConf()
	f(in1.Pointer(), in2.Pointer(), in3.Pointer(), out.Pointer(), init, N, gr, bl)
	return copyback(out)
}

// general reduce wrapper for 6 input arrays
func reduce6(in1, in2, in3, in4, in5, in6 safe.Float32s, init float32, f func(in1, in2, in3, in4, in5, in6, out cu.DevicePtr, init float32, N int, grid, block cu.Dim3)) float32 {
	N := in1.Len()
	core.Assert(in2.Len() == N && in3.Len() == N && in4.Len() == N && in5.Len() == N && in6.Len() == N)
	out := reduceBuf(init)
	defer reduceRecycle(out)
	gr, bl := reduceConf()
	f(in1.Pointer(), in2.Pointer(), in3.Pointer(), in4.Pointer(), in5.Pointer(), in6.Pointer(), out.Pointer(), init, N, gr, bl)
	return copyback(out)
}

// Maximum difference between the two arrays.
// 	max_i abs(a[i] - b[i])
func MaxDiff(a, b safe.Float32s, stream cu.Stream) float32 {
	return reduce2(a, b, 0, ptx.K_reducemaxdiff)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(x, y, z safe.Float32s, stream cu.Stream) float64 {
	r := reduce3(x, y, z, 0, ptx.K_reducemaxvecnorm2)
	return math.Sqrt(float64(r))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x1, y1, z1, x2, y2, z2 safe.Float32s, stream cu.Stream) float64 {
	r := reduce6(x1, y1, z1, x2, y2, z2, 0, ptx.K_reducemaxvecdiff2)
	return math.Sqrt(float64(r))
}

// Vector dot product.
func Dot(x1, x2 safe.Float32s) float32 {
	return reduce2(x1, x2, 0, ptx.K_reducedot)
}

var reduceBuffers chan safe.Float32s // pool of 1-float CUDA buffers for reduce

// recycle a 1-float CUDA reduction buffer
func reduceRecycle(buf safe.Float32s) {
	reduceBuffers <- buf
}

// return a 1-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) safe.Float32s {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	buf.Memset(initVal)
	return buf
}

func copyback(buf safe.Float32s) float32 {
	var result_ [1]float32
	result := result_[:]
	buf.CopyDtoH(result) // async? register one arena block // , stream)
	return result_[0]
}

// initialize pool of 1-float CUDA reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan safe.Float32s, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- safe.MakeFloat32s(1)
	}
}

// launch configuration for reduce kernels
func reduceConf() (gridDim, blockDim cu.Dim3) {
	blockDim = cu.Dim3{512, 1, 1}
	gridDim = cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.
	// could be improved but takes hardly ~1% of execution time
	return
}
