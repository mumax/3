package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"math"
	"unsafe"
)

// Sum of all elements.
func Sum(in safe.Float32s, stream cu.Stream) float32 {
	return reduce("reducesum", in, stream)
}

// Maximum of all elements.
func Max(in safe.Float32s, stream cu.Stream) float32 {
	return reduce("reducemax", in, stream)
}

// Minimum of all elements.
func Min(in safe.Float32s, stream cu.Stream) float32 {
	return reduce("reducemin", in, stream)
}

// Maximum of absolute values of all elements.
func MaxAbs(in safe.Float32s, stream cu.Stream) float32 {
	return reduce("reducemaxabs", in, stream)
}

// Maximum difference between the two arrays.
// 	max_i abs(a[i] - b[i])
func MaxDiff(a, b safe.Float32s, stream cu.Stream) float32 {
	return reduce2("reducemaxdiff", a, b, stream)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
// 	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(x, y, z safe.Float32s, stream cu.Stream) float64 {
	return math.Sqrt(float64(reduce3("reducemaxvecnorm2", x, y, z, stream)))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
// 	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
// 	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x1, y1, z1, x2, y2, z2 safe.Float32s, stream cu.Stream) float64 {
	return math.Sqrt(float64(reduce6("reducemaxvecdiff2", x1, y1, z1, x2, y2, z2, stream)))
}

// 1-input reduce with arbitrary PTX code (op)
func reduce(op string, in safe.Float32s, stream cu.Stream) float32 {
	out := safe.MakeFloat32s(1)
	defer out.Free()

	N := in.Len()

	blockDim := cu.Dim3{512, 1, 1}
	gridDim := cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.

	//ptxcallAsync(op, gridDim, blockDim, 0, stream, in.Pointer(), out.Pointer(), N)
	inptr := in.Pointer()
	outptr := out.Pointer()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inptr),
		unsafe.Pointer(&outptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad(op)
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()

	var result_ [1]float32
	result := result_[:]
	out.CopyDtoH(result) // async? register one arena block // , stream)
	return result_[0]
}

// 2-input reduce with arbitrary PTX code (op)
func reduce2(op string, in1, in2 safe.Float32s, stream cu.Stream) float32 {
	out := safe.MakeFloat32s(1)
	defer out.Free()

	core.Assert(in1.Len() == in2.Len())
	N := in1.Len()

	blockDim := cu.Dim3{512, 1, 1}
	gridDim := cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.

	in1ptr := in1.Pointer()
	in2ptr := in2.Pointer()
	outptr := out.Pointer()
	args := []unsafe.Pointer{
		unsafe.Pointer(&in1ptr),
		unsafe.Pointer(&in2ptr),
		unsafe.Pointer(&outptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad(op)
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()

	var result_ [1]float32
	result := result_[:]
	out.CopyDtoH(result) // async? register one arena block // , stream)
	return result_[0]
}

// 3-input reduce with arbitrary PTX code (op)
func reduce3(op string, in1, in2, in3 safe.Float32s, stream cu.Stream) float32 {
	out := safe.MakeFloat32s(1)
	defer out.Free()

	core.Assert(in1.Len() == in2.Len())
	N := in1.Len()

	blockDim := cu.Dim3{512, 1, 1}
	gridDim := cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.

	in1ptr := in1.Pointer()
	in2ptr := in2.Pointer()
	in3ptr := in3.Pointer()
	outptr := out.Pointer()
	args := []unsafe.Pointer{
		unsafe.Pointer(&in1ptr),
		unsafe.Pointer(&in2ptr),
		unsafe.Pointer(&in3ptr),
		unsafe.Pointer(&outptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad(op)
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()

	var result_ [1]float32
	result := result_[:]
	out.CopyDtoH(result) // async? register one arena block // , stream)
	return result_[0]
}

// 6-input reduce with arbitrary PTX code (op)
func reduce6(op string, in1, in2, in3, in4, in5, in6 safe.Float32s, stream cu.Stream) float32 {
	out := safe.MakeFloat32s(1)
	defer out.Free()

	core.Assert(in1.Len() == in2.Len())
	N := in1.Len()

	blockDim := cu.Dim3{512, 1, 1}
	gridDim := cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.

	in1ptr := in1.Pointer()
	in2ptr := in2.Pointer()
	in3ptr := in3.Pointer()
	in4ptr := in4.Pointer()
	in5ptr := in5.Pointer()
	in6ptr := in6.Pointer()
	outptr := out.Pointer()
	args := []unsafe.Pointer{
		unsafe.Pointer(&in1ptr),
		unsafe.Pointer(&in2ptr),
		unsafe.Pointer(&in3ptr),
		unsafe.Pointer(&in4ptr),
		unsafe.Pointer(&in5ptr),
		unsafe.Pointer(&in6ptr),
		unsafe.Pointer(&outptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad(op)
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()

	var result_ [1]float32
	result := result_[:]
	out.CopyDtoH(result) // async? register one arena block // , stream)
	return result_[0]
}

//func ptxcallAsync(code string, gridDim, blockDim cu.Dim3, shmem int, stream cu.Stream, args ...interface{}) {
//	argptr := make([]unsafe.Pointer, len(args))
//	for i, arg := range args {
//		switch v := arg.(type) {
//		default:
//			panic(fmt.Errorf("ptxcall: unsupported type: %v", reflect.TypeOf(arg)))
//		case int:
//			argptr[i] = unsafe.Pointer(&v)
//			fmt.Println("arg", i, *((*int)(argptr[i])))
//		case cu.DevicePtr, uintptr:
//			argptr[i] = unsafe.Pointer(&v)
//		}
//	}
//	kernel := PTXLoad(code)
//	cu.LaunchKernel(kernel, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, argptr)
//}
