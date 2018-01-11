package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for shifty kernel
var shifty_code cu.Function

// Stores the arguments for shifty kernel invocation
type shifty_args_t struct {
	arg_dst    unsafe.Pointer
	arg_src    unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	arg_shy    int
	arg_clampL float32
	arg_clampR float32
	argptr     [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for shifty kernel invocation
var shifty_args shifty_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	shifty_args.argptr[0] = unsafe.Pointer(&shifty_args.arg_dst)
	shifty_args.argptr[1] = unsafe.Pointer(&shifty_args.arg_src)
	shifty_args.argptr[2] = unsafe.Pointer(&shifty_args.arg_Nx)
	shifty_args.argptr[3] = unsafe.Pointer(&shifty_args.arg_Ny)
	shifty_args.argptr[4] = unsafe.Pointer(&shifty_args.arg_Nz)
	shifty_args.argptr[5] = unsafe.Pointer(&shifty_args.arg_shy)
	shifty_args.argptr[6] = unsafe.Pointer(&shifty_args.arg_clampL)
	shifty_args.argptr[7] = unsafe.Pointer(&shifty_args.arg_clampR)
}

// Wrapper for shifty CUDA kernel, asynchronous.
func k_shifty_async(dst unsafe.Pointer, src unsafe.Pointer, Nx int, Ny int, Nz int, shy int, clampL float32, clampR float32, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("shifty")
	}

	shifty_args.Lock()
	defer shifty_args.Unlock()

	if shifty_code == 0 {
		shifty_code = fatbinLoad(shifty_map, "shifty")
	}

	shifty_args.arg_dst = dst
	shifty_args.arg_src = src
	shifty_args.arg_Nx = Nx
	shifty_args.arg_Ny = Ny
	shifty_args.arg_Nz = Nz
	shifty_args.arg_shy = shy
	shifty_args.arg_clampL = clampL
	shifty_args.arg_clampR = clampR

	args := shifty_args.argptr[:]
	cu.LaunchKernel(shifty_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("shifty")
	}
}

// maps compute capability on PTX code for shifty kernel.
var shifty_map = map[int]string{0: "",
	20: shifty_ptx_20,
	30: shifty_ptx_30,
	35: shifty_ptx_35,
	50: shifty_ptx_50,
	52: shifty_ptx_52,
	53: shifty_ptx_53,
	60: shifty_ptx_60,
	61: shifty_ptx_61,
	62: shifty_ptx_62,
	70: shifty_ptx_70}

// shifty PTX code for various compute capabilities.
const (
	shifty_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<22>;
	.reg .f32 	%f<6>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shifty_param_0];
	ld.param.u64 	%rd4, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 1 9 1
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	.loc 1 10 1
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	.loc 1 11 1
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	.loc 1 13 1
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	.loc 1 13 1
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	.loc 1 13 1
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	.loc 1 14 1
	sub.s32 	%r4, %r2, %r7;
	.loc 1 16 1
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB0_4;

	.loc 1 18 1
	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB0_4;

	.loc 1 21 1
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd5, %r19, 4;
	add.s64 	%rd6, %rd2, %rd5;
	.loc 1 21 1
	ld.global.f32 	%f5, [%rd6];

BB0_4:
	.loc 1 23 1
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd1, %rd7;
	.loc 1 23 1
	st.global.f32 	[%rd8], %f5;

BB0_5:
	.loc 1 25 2
	ret;
}


`
	shifty_ptx_30 = `
.version 4.0
.target sm_30
.address_size 64


.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<22>;
	.reg .f32 	%f<6>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB0_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f5, [%rd5];

BB0_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB0_5:
	ret;
}


`
	shifty_ptx_35 = `
.version 4.1
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<22>;
	.reg .f32 	%f<6>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB5_5;
	bra.uni 	BB5_1;

BB5_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB5_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB5_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB5_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB5_5:
	ret;
}


`
	shifty_ptx_50 = `
.version 4.3
.target sm_50
.address_size 64

	// .weak	cudaMalloc

.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaFuncGetAttributes
.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaDeviceGetAttribute
.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaGetDevice
.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessor
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .globl	shifty
.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB6_5;
	bra.uni 	BB6_1;

BB6_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB6_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB6_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB6_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB6_5:
	ret;
}


`
	shifty_ptx_52 = `
.version 4.3
.target sm_52
.address_size 64

	// .weak	cudaMalloc

.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaFuncGetAttributes
.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaDeviceGetAttribute
.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaGetDevice
.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessor
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .globl	shifty
.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB6_5;
	bra.uni 	BB6_1;

BB6_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB6_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB6_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB6_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB6_5:
	ret;
}


`
	shifty_ptx_53 = `
.version 4.3
.target sm_53
.address_size 64

	// .weak	cudaMalloc

.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaFuncGetAttributes
.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaDeviceGetAttribute
.weak .func  (.param .b32 func_retval0) cudaDeviceGetAttribute(
	.param .b64 cudaDeviceGetAttribute_param_0,
	.param .b32 cudaDeviceGetAttribute_param_1,
	.param .b32 cudaDeviceGetAttribute_param_2
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaGetDevice
.weak .func  (.param .b32 func_retval0) cudaGetDevice(
	.param .b64 cudaGetDevice_param_0
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessor
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessor_param_3
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .weak	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.weak .func  (.param .b32 func_retval0) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_0,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_1,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_2,
	.param .b64 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_3,
	.param .b32 cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_param_4
)
{
	.reg .b32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

	// .globl	shifty
.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB6_5;
	bra.uni 	BB6_1;

BB6_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB6_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB6_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB6_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB6_5:
	ret;
}


`
	shifty_ptx_60 = `
.version 5.0
.target sm_60
.address_size 64

	// .globl	shifty

.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB0_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB0_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB0_5:
	ret;
}


`
	shifty_ptx_61 = `
.version 5.0
.target sm_61
.address_size 64

	// .globl	shifty

.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB0_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB0_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB0_5:
	ret;
}


`
	shifty_ptx_62 = `
.version 5.0
.target sm_62
.address_size 64

	// .globl	shifty

.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f3, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	mov.f32 	%f5, %f3;
	@%p6 bra 	BB0_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB0_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB0_5:
	ret;
}


`
	shifty_ptx_70 = `
.version 6.0
.target sm_70
.address_size 64

	// .globl	shifty

.visible .entry shifty(
	.param .u64 shifty_param_0,
	.param .u64 shifty_param_1,
	.param .u32 shifty_param_2,
	.param .u32 shifty_param_3,
	.param .u32 shifty_param_4,
	.param .u32 shifty_param_5,
	.param .f32 shifty_param_6,
	.param .f32 shifty_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [shifty_param_0];
	ld.param.u64 	%rd2, [shifty_param_1];
	ld.param.u32 	%r5, [shifty_param_2];
	ld.param.u32 	%r6, [shifty_param_3];
	ld.param.u32 	%r8, [shifty_param_4];
	ld.param.u32 	%r7, [shifty_param_5];
	ld.param.f32 	%f5, [shifty_param_6];
	ld.param.f32 	%f4, [shifty_param_7];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r2, %r6;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	sub.s32 	%r4, %r2, %r7;
	setp.lt.s32	%p6, %r4, 0;
	@%p6 bra 	BB0_4;

	setp.ge.s32	%p7, %r4, %r6;
	mov.f32 	%f5, %f4;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r6, %r4;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f5, [%rd5];

BB0_4:
	cvta.to.global.u64 	%rd6, %rd1;
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB0_5:
	ret;
}


`
)
