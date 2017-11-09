package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
)

// CUDA handle for settemperatureJH kernel
var settemperatureJH_code cu.Function

// Stores the arguments for settemperatureJH kernel invocation
type settemperatureJH_args_t struct{
	 arg_B unsafe.Pointer
	 arg_noise unsafe.Pointer
	 arg_kB2_VgammaDt float32
	 arg_Ms_ unsafe.Pointer
	 arg_Ms_mul float32
	 arg_tempJH unsafe.Pointer
	 arg_alpha_ unsafe.Pointer
	 arg_alpha_mul float32
	 arg_N int
	 argptr [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for settemperatureJH kernel invocation
var settemperatureJH_args settemperatureJH_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 settemperatureJH_args.argptr[0] = unsafe.Pointer(&settemperatureJH_args.arg_B)
	 settemperatureJH_args.argptr[1] = unsafe.Pointer(&settemperatureJH_args.arg_noise)
	 settemperatureJH_args.argptr[2] = unsafe.Pointer(&settemperatureJH_args.arg_kB2_VgammaDt)
	 settemperatureJH_args.argptr[3] = unsafe.Pointer(&settemperatureJH_args.arg_Ms_)
	 settemperatureJH_args.argptr[4] = unsafe.Pointer(&settemperatureJH_args.arg_Ms_mul)
	 settemperatureJH_args.argptr[5] = unsafe.Pointer(&settemperatureJH_args.arg_tempJH)
	 settemperatureJH_args.argptr[6] = unsafe.Pointer(&settemperatureJH_args.arg_alpha_)
	 settemperatureJH_args.argptr[7] = unsafe.Pointer(&settemperatureJH_args.arg_alpha_mul)
	 settemperatureJH_args.argptr[8] = unsafe.Pointer(&settemperatureJH_args.arg_N)
	 }

// Wrapper for settemperatureJH CUDA kernel, asynchronous.
func k_settemperatureJH_async ( B unsafe.Pointer, noise unsafe.Pointer, kB2_VgammaDt float32, Ms_ unsafe.Pointer, Ms_mul float32, tempJH unsafe.Pointer, alpha_ unsafe.Pointer, alpha_mul float32, N int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("settemperatureJH")
	}

	settemperatureJH_args.Lock()
	defer settemperatureJH_args.Unlock()

	if settemperatureJH_code == 0{
		settemperatureJH_code = fatbinLoad(settemperatureJH_map, "settemperatureJH")
	}

	 settemperatureJH_args.arg_B = B
	 settemperatureJH_args.arg_noise = noise
	 settemperatureJH_args.arg_kB2_VgammaDt = kB2_VgammaDt
	 settemperatureJH_args.arg_Ms_ = Ms_
	 settemperatureJH_args.arg_Ms_mul = Ms_mul
	 settemperatureJH_args.arg_tempJH = tempJH
	 settemperatureJH_args.arg_alpha_ = alpha_
	 settemperatureJH_args.arg_alpha_mul = alpha_mul
	 settemperatureJH_args.arg_N = N
	

	args := settemperatureJH_args.argptr[:]
	cu.LaunchKernel(settemperatureJH_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("settemperatureJH")
	}
}

// maps compute capability on PTX code for settemperatureJH kernel.
var settemperatureJH_map = map[int]string{ 0: "" ,
20: settemperatureJH_ptx_20 ,
30: settemperatureJH_ptx_30 ,
35: settemperatureJH_ptx_35 ,
50: settemperatureJH_ptx_50 ,
52: settemperatureJH_ptx_52 ,
53: settemperatureJH_ptx_53  }

// settemperatureJH PTX code for various compute capabilities.
const(
  settemperatureJH_ptx_20 = `
.version 4.3
.target sm_20
.address_size 64

	// .globl	_Z8inv_MsatPffi

.visible .func  (.param .b32 func_retval0) _Z8inv_MsatPffi(
	.param .b64 _Z8inv_MsatPffi_param_0,
	.param .b32 _Z8inv_MsatPffi_param_1,
	.param .b32 _Z8inv_MsatPffi_param_2
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z8inv_MsatPffi_param_0];
	ld.param.f32 	%f8, [_Z8inv_MsatPffi_param_1];
	ld.param.u32 	%r1, [_Z8inv_MsatPffi_param_2];
	setp.eq.s64	%p1, %rd1, 0;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	ld.f32 	%f6, [%rd3];
	mul.f32 	%f8, %f6, %f8;

BB0_2:
	setp.eq.f32	%p2, %f8, 0f00000000;
	mov.f32 	%f9, 0f00000000;
	@%p2 bra 	BB0_4;

	rcp.rn.f32 	%f9, %f8;

BB0_4:
	st.param.f32	[func_retval0+0], %f9;
	ret;
}

	// .globl	settemperatureJH
.visible .entry settemperatureJH(
	.param .u64 settemperatureJH_param_0,
	.param .u64 settemperatureJH_param_1,
	.param .f32 settemperatureJH_param_2,
	.param .u64 settemperatureJH_param_3,
	.param .f32 settemperatureJH_param_4,
	.param .u64 settemperatureJH_param_5,
	.param .u64 settemperatureJH_param_6,
	.param .f32 settemperatureJH_param_7,
	.param .u32 settemperatureJH_param_8
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd2, [settemperatureJH_param_0];
	ld.param.u64 	%rd3, [settemperatureJH_param_1];
	ld.param.f32 	%f8, [settemperatureJH_param_2];
	ld.param.u64 	%rd4, [settemperatureJH_param_3];
	ld.param.f32 	%f20, [settemperatureJH_param_4];
	ld.param.u64 	%rd5, [settemperatureJH_param_5];
	ld.param.u64 	%rd6, [settemperatureJH_param_6];
	ld.param.f32 	%f22, [settemperatureJH_param_7];
	ld.param.u32 	%r2, [settemperatureJH_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB1_8;

	setp.eq.s64	%p2, %rd4, 0;
	@%p2 bra 	BB1_3;

	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f11, [%rd9];
	mul.f32 	%f20, %f11, %f20;

BB1_3:
	setp.eq.f32	%p3, %f20, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	BB1_5;

	rcp.rn.f32 	%f21, %f20;

BB1_5:
	cvta.to.global.u64 	%rd10, %rd5;
	cvt.s64.s32	%rd1, %r1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f5, [%rd12];
	setp.eq.s64	%p4, %rd6, 0;
	@%p4 bra 	BB1_7;

	cvta.to.global.u64 	%rd13, %rd6;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.f32 	%f13, [%rd15];
	mul.f32 	%f22, %f13, %f22;

BB1_7:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	mul.f32 	%f14, %f22, %f8;
	mul.f32 	%f15, %f5, %f14;
	mul.f32 	%f16, %f21, %f15;
	sqrt.rn.f32 	%f17, %f16;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f32 	%f18, [%rd19];
	mul.f32 	%f19, %f18, %f17;
	add.s64 	%rd20, %rd16, %rd18;
	st.global.f32 	[%rd20], %f19;

BB1_8:
	ret;
}


`
   settemperatureJH_ptx_30 = `
.version 4.3
.target sm_30
.address_size 64

	// .globl	_Z8inv_MsatPffi

.visible .func  (.param .b32 func_retval0) _Z8inv_MsatPffi(
	.param .b64 _Z8inv_MsatPffi_param_0,
	.param .b32 _Z8inv_MsatPffi_param_1,
	.param .b32 _Z8inv_MsatPffi_param_2
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z8inv_MsatPffi_param_0];
	ld.param.f32 	%f8, [_Z8inv_MsatPffi_param_1];
	ld.param.u32 	%r1, [_Z8inv_MsatPffi_param_2];
	setp.eq.s64	%p1, %rd1, 0;
	@%p1 bra 	BB0_2;

	mul.wide.s32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	ld.f32 	%f6, [%rd3];
	mul.f32 	%f8, %f6, %f8;

BB0_2:
	setp.eq.f32	%p2, %f8, 0f00000000;
	mov.f32 	%f9, 0f00000000;
	@%p2 bra 	BB0_4;

	rcp.rn.f32 	%f9, %f8;

BB0_4:
	st.param.f32	[func_retval0+0], %f9;
	ret;
}

	// .globl	settemperatureJH
.visible .entry settemperatureJH(
	.param .u64 settemperatureJH_param_0,
	.param .u64 settemperatureJH_param_1,
	.param .f32 settemperatureJH_param_2,
	.param .u64 settemperatureJH_param_3,
	.param .f32 settemperatureJH_param_4,
	.param .u64 settemperatureJH_param_5,
	.param .u64 settemperatureJH_param_6,
	.param .f32 settemperatureJH_param_7,
	.param .u32 settemperatureJH_param_8
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd2, [settemperatureJH_param_0];
	ld.param.u64 	%rd3, [settemperatureJH_param_1];
	ld.param.f32 	%f8, [settemperatureJH_param_2];
	ld.param.u64 	%rd4, [settemperatureJH_param_3];
	ld.param.f32 	%f20, [settemperatureJH_param_4];
	ld.param.u64 	%rd5, [settemperatureJH_param_5];
	ld.param.u64 	%rd6, [settemperatureJH_param_6];
	ld.param.f32 	%f22, [settemperatureJH_param_7];
	ld.param.u32 	%r2, [settemperatureJH_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB1_8;

	setp.eq.s64	%p2, %rd4, 0;
	@%p2 bra 	BB1_3;

	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f11, [%rd9];
	mul.f32 	%f20, %f11, %f20;

BB1_3:
	setp.eq.f32	%p3, %f20, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	BB1_5;

	rcp.rn.f32 	%f21, %f20;

BB1_5:
	cvta.to.global.u64 	%rd10, %rd5;
	cvt.s64.s32	%rd1, %r1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f5, [%rd12];
	setp.eq.s64	%p4, %rd6, 0;
	@%p4 bra 	BB1_7;

	cvta.to.global.u64 	%rd13, %rd6;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.f32 	%f13, [%rd15];
	mul.f32 	%f22, %f13, %f22;

BB1_7:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	mul.f32 	%f14, %f22, %f8;
	mul.f32 	%f15, %f5, %f14;
	mul.f32 	%f16, %f21, %f15;
	sqrt.rn.f32 	%f17, %f16;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f32 	%f18, [%rd19];
	mul.f32 	%f19, %f18, %f17;
	add.s64 	%rd20, %rd16, %rd18;
	st.global.f32 	[%rd20], %f19;

BB1_8:
	ret;
}


`
   settemperatureJH_ptx_35 = `
.version 4.3
.target sm_35
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

	// .globl	_Z8inv_MsatPffi
.visible .func  (.param .b32 func_retval0) _Z8inv_MsatPffi(
	.param .b64 _Z8inv_MsatPffi_param_0,
	.param .b32 _Z8inv_MsatPffi_param_1,
	.param .b32 _Z8inv_MsatPffi_param_2
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z8inv_MsatPffi_param_0];
	ld.param.f32 	%f8, [_Z8inv_MsatPffi_param_1];
	ld.param.u32 	%r1, [_Z8inv_MsatPffi_param_2];
	setp.eq.s64	%p1, %rd1, 0;
	@%p1 bra 	BB6_2;

	mul.wide.s32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	ld.f32 	%f6, [%rd3];
	mul.f32 	%f8, %f6, %f8;

BB6_2:
	setp.eq.f32	%p2, %f8, 0f00000000;
	mov.f32 	%f9, 0f00000000;
	@%p2 bra 	BB6_4;

	rcp.rn.f32 	%f9, %f8;

BB6_4:
	st.param.f32	[func_retval0+0], %f9;
	ret;
}

	// .globl	settemperatureJH
.visible .entry settemperatureJH(
	.param .u64 settemperatureJH_param_0,
	.param .u64 settemperatureJH_param_1,
	.param .f32 settemperatureJH_param_2,
	.param .u64 settemperatureJH_param_3,
	.param .f32 settemperatureJH_param_4,
	.param .u64 settemperatureJH_param_5,
	.param .u64 settemperatureJH_param_6,
	.param .f32 settemperatureJH_param_7,
	.param .u32 settemperatureJH_param_8
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd2, [settemperatureJH_param_0];
	ld.param.u64 	%rd3, [settemperatureJH_param_1];
	ld.param.f32 	%f8, [settemperatureJH_param_2];
	ld.param.u64 	%rd4, [settemperatureJH_param_3];
	ld.param.f32 	%f20, [settemperatureJH_param_4];
	ld.param.u64 	%rd5, [settemperatureJH_param_5];
	ld.param.u64 	%rd6, [settemperatureJH_param_6];
	ld.param.f32 	%f22, [settemperatureJH_param_7];
	ld.param.u32 	%r2, [settemperatureJH_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB7_8;

	setp.eq.s64	%p2, %rd4, 0;
	@%p2 bra 	BB7_3;

	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f11, [%rd9];
	mul.f32 	%f20, %f11, %f20;

BB7_3:
	setp.eq.f32	%p3, %f20, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	BB7_5;

	rcp.rn.f32 	%f21, %f20;

BB7_5:
	cvta.to.global.u64 	%rd10, %rd5;
	cvt.s64.s32	%rd1, %r1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f5, [%rd12];
	setp.eq.s64	%p4, %rd6, 0;
	@%p4 bra 	BB7_7;

	cvta.to.global.u64 	%rd13, %rd6;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f13, [%rd15];
	mul.f32 	%f22, %f13, %f22;

BB7_7:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	mul.f32 	%f14, %f22, %f8;
	mul.f32 	%f15, %f5, %f14;
	mul.f32 	%f16, %f21, %f15;
	sqrt.rn.f32 	%f17, %f16;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.nc.f32 	%f18, [%rd19];
	mul.f32 	%f19, %f18, %f17;
	add.s64 	%rd20, %rd16, %rd18;
	st.global.f32 	[%rd20], %f19;

BB7_8:
	ret;
}


`
   settemperatureJH_ptx_50 = `
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

	// .globl	_Z8inv_MsatPffi
.visible .func  (.param .b32 func_retval0) _Z8inv_MsatPffi(
	.param .b64 _Z8inv_MsatPffi_param_0,
	.param .b32 _Z8inv_MsatPffi_param_1,
	.param .b32 _Z8inv_MsatPffi_param_2
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z8inv_MsatPffi_param_0];
	ld.param.f32 	%f8, [_Z8inv_MsatPffi_param_1];
	ld.param.u32 	%r1, [_Z8inv_MsatPffi_param_2];
	setp.eq.s64	%p1, %rd1, 0;
	@%p1 bra 	BB6_2;

	mul.wide.s32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	ld.f32 	%f6, [%rd3];
	mul.f32 	%f8, %f6, %f8;

BB6_2:
	setp.eq.f32	%p2, %f8, 0f00000000;
	mov.f32 	%f9, 0f00000000;
	@%p2 bra 	BB6_4;

	rcp.rn.f32 	%f9, %f8;

BB6_4:
	st.param.f32	[func_retval0+0], %f9;
	ret;
}

	// .globl	settemperatureJH
.visible .entry settemperatureJH(
	.param .u64 settemperatureJH_param_0,
	.param .u64 settemperatureJH_param_1,
	.param .f32 settemperatureJH_param_2,
	.param .u64 settemperatureJH_param_3,
	.param .f32 settemperatureJH_param_4,
	.param .u64 settemperatureJH_param_5,
	.param .u64 settemperatureJH_param_6,
	.param .f32 settemperatureJH_param_7,
	.param .u32 settemperatureJH_param_8
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd2, [settemperatureJH_param_0];
	ld.param.u64 	%rd3, [settemperatureJH_param_1];
	ld.param.f32 	%f8, [settemperatureJH_param_2];
	ld.param.u64 	%rd4, [settemperatureJH_param_3];
	ld.param.f32 	%f20, [settemperatureJH_param_4];
	ld.param.u64 	%rd5, [settemperatureJH_param_5];
	ld.param.u64 	%rd6, [settemperatureJH_param_6];
	ld.param.f32 	%f22, [settemperatureJH_param_7];
	ld.param.u32 	%r2, [settemperatureJH_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB7_8;

	setp.eq.s64	%p2, %rd4, 0;
	@%p2 bra 	BB7_3;

	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f11, [%rd9];
	mul.f32 	%f20, %f11, %f20;

BB7_3:
	setp.eq.f32	%p3, %f20, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	BB7_5;

	rcp.rn.f32 	%f21, %f20;

BB7_5:
	cvta.to.global.u64 	%rd10, %rd5;
	cvt.s64.s32	%rd1, %r1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f5, [%rd12];
	setp.eq.s64	%p4, %rd6, 0;
	@%p4 bra 	BB7_7;

	cvta.to.global.u64 	%rd13, %rd6;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f13, [%rd15];
	mul.f32 	%f22, %f13, %f22;

BB7_7:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	mul.f32 	%f14, %f22, %f8;
	mul.f32 	%f15, %f5, %f14;
	mul.f32 	%f16, %f21, %f15;
	sqrt.rn.f32 	%f17, %f16;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.nc.f32 	%f18, [%rd19];
	mul.f32 	%f19, %f18, %f17;
	add.s64 	%rd20, %rd16, %rd18;
	st.global.f32 	[%rd20], %f19;

BB7_8:
	ret;
}


`
   settemperatureJH_ptx_52 = `
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

	// .globl	_Z8inv_MsatPffi
.visible .func  (.param .b32 func_retval0) _Z8inv_MsatPffi(
	.param .b64 _Z8inv_MsatPffi_param_0,
	.param .b32 _Z8inv_MsatPffi_param_1,
	.param .b32 _Z8inv_MsatPffi_param_2
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z8inv_MsatPffi_param_0];
	ld.param.f32 	%f8, [_Z8inv_MsatPffi_param_1];
	ld.param.u32 	%r1, [_Z8inv_MsatPffi_param_2];
	setp.eq.s64	%p1, %rd1, 0;
	@%p1 bra 	BB6_2;

	mul.wide.s32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	ld.f32 	%f6, [%rd3];
	mul.f32 	%f8, %f6, %f8;

BB6_2:
	setp.eq.f32	%p2, %f8, 0f00000000;
	mov.f32 	%f9, 0f00000000;
	@%p2 bra 	BB6_4;

	rcp.rn.f32 	%f9, %f8;

BB6_4:
	st.param.f32	[func_retval0+0], %f9;
	ret;
}

	// .globl	settemperatureJH
.visible .entry settemperatureJH(
	.param .u64 settemperatureJH_param_0,
	.param .u64 settemperatureJH_param_1,
	.param .f32 settemperatureJH_param_2,
	.param .u64 settemperatureJH_param_3,
	.param .f32 settemperatureJH_param_4,
	.param .u64 settemperatureJH_param_5,
	.param .u64 settemperatureJH_param_6,
	.param .f32 settemperatureJH_param_7,
	.param .u32 settemperatureJH_param_8
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd2, [settemperatureJH_param_0];
	ld.param.u64 	%rd3, [settemperatureJH_param_1];
	ld.param.f32 	%f8, [settemperatureJH_param_2];
	ld.param.u64 	%rd4, [settemperatureJH_param_3];
	ld.param.f32 	%f20, [settemperatureJH_param_4];
	ld.param.u64 	%rd5, [settemperatureJH_param_5];
	ld.param.u64 	%rd6, [settemperatureJH_param_6];
	ld.param.f32 	%f22, [settemperatureJH_param_7];
	ld.param.u32 	%r2, [settemperatureJH_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB7_8;

	setp.eq.s64	%p2, %rd4, 0;
	@%p2 bra 	BB7_3;

	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f11, [%rd9];
	mul.f32 	%f20, %f11, %f20;

BB7_3:
	setp.eq.f32	%p3, %f20, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	BB7_5;

	rcp.rn.f32 	%f21, %f20;

BB7_5:
	cvta.to.global.u64 	%rd10, %rd5;
	cvt.s64.s32	%rd1, %r1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f5, [%rd12];
	setp.eq.s64	%p4, %rd6, 0;
	@%p4 bra 	BB7_7;

	cvta.to.global.u64 	%rd13, %rd6;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f13, [%rd15];
	mul.f32 	%f22, %f13, %f22;

BB7_7:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	mul.f32 	%f14, %f22, %f8;
	mul.f32 	%f15, %f5, %f14;
	mul.f32 	%f16, %f21, %f15;
	sqrt.rn.f32 	%f17, %f16;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.nc.f32 	%f18, [%rd19];
	mul.f32 	%f19, %f18, %f17;
	add.s64 	%rd20, %rd16, %rd18;
	st.global.f32 	[%rd20], %f19;

BB7_8:
	ret;
}


`
   settemperatureJH_ptx_53 = `
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

	// .globl	_Z8inv_MsatPffi
.visible .func  (.param .b32 func_retval0) _Z8inv_MsatPffi(
	.param .b64 _Z8inv_MsatPffi_param_0,
	.param .b32 _Z8inv_MsatPffi_param_1,
	.param .b32 _Z8inv_MsatPffi_param_2
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z8inv_MsatPffi_param_0];
	ld.param.f32 	%f8, [_Z8inv_MsatPffi_param_1];
	ld.param.u32 	%r1, [_Z8inv_MsatPffi_param_2];
	setp.eq.s64	%p1, %rd1, 0;
	@%p1 bra 	BB6_2;

	mul.wide.s32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	ld.f32 	%f6, [%rd3];
	mul.f32 	%f8, %f6, %f8;

BB6_2:
	setp.eq.f32	%p2, %f8, 0f00000000;
	mov.f32 	%f9, 0f00000000;
	@%p2 bra 	BB6_4;

	rcp.rn.f32 	%f9, %f8;

BB6_4:
	st.param.f32	[func_retval0+0], %f9;
	ret;
}

	// .globl	settemperatureJH
.visible .entry settemperatureJH(
	.param .u64 settemperatureJH_param_0,
	.param .u64 settemperatureJH_param_1,
	.param .f32 settemperatureJH_param_2,
	.param .u64 settemperatureJH_param_3,
	.param .f32 settemperatureJH_param_4,
	.param .u64 settemperatureJH_param_5,
	.param .u64 settemperatureJH_param_6,
	.param .f32 settemperatureJH_param_7,
	.param .u32 settemperatureJH_param_8
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd2, [settemperatureJH_param_0];
	ld.param.u64 	%rd3, [settemperatureJH_param_1];
	ld.param.f32 	%f8, [settemperatureJH_param_2];
	ld.param.u64 	%rd4, [settemperatureJH_param_3];
	ld.param.f32 	%f20, [settemperatureJH_param_4];
	ld.param.u64 	%rd5, [settemperatureJH_param_5];
	ld.param.u64 	%rd6, [settemperatureJH_param_6];
	ld.param.f32 	%f22, [settemperatureJH_param_7];
	ld.param.u32 	%r2, [settemperatureJH_param_8];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB7_8;

	setp.eq.s64	%p2, %rd4, 0;
	@%p2 bra 	BB7_3;

	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f11, [%rd9];
	mul.f32 	%f20, %f11, %f20;

BB7_3:
	setp.eq.f32	%p3, %f20, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	BB7_5;

	rcp.rn.f32 	%f21, %f20;

BB7_5:
	cvta.to.global.u64 	%rd10, %rd5;
	cvt.s64.s32	%rd1, %r1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f5, [%rd12];
	setp.eq.s64	%p4, %rd6, 0;
	@%p4 bra 	BB7_7;

	cvta.to.global.u64 	%rd13, %rd6;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f13, [%rd15];
	mul.f32 	%f22, %f13, %f22;

BB7_7:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	mul.f32 	%f14, %f22, %f8;
	mul.f32 	%f15, %f5, %f14;
	mul.f32 	%f16, %f21, %f15;
	sqrt.rn.f32 	%f17, %f16;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.nc.f32 	%f18, [%rd19];
	mul.f32 	%f19, %f18, %f17;
	add.s64 	%rd20, %rd16, %rd18;
	st.global.f32 	[%rd20], %f19;

BB7_8:
	ret;
}


`
 )
