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

// CUDA handle for reducemaxdiff kernel
var reducemaxdiff_code cu.Function

// Stores the arguments for reducemaxdiff kernel invocation
type reducemaxdiff_args_t struct{
	 arg_src1 unsafe.Pointer
	 arg_src2 unsafe.Pointer
	 arg_dst unsafe.Pointer
	 arg_initVal float32
	 arg_n int
	 argptr [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducemaxdiff kernel invocation
var reducemaxdiff_args reducemaxdiff_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 reducemaxdiff_args.argptr[0] = unsafe.Pointer(&reducemaxdiff_args.arg_src1)
	 reducemaxdiff_args.argptr[1] = unsafe.Pointer(&reducemaxdiff_args.arg_src2)
	 reducemaxdiff_args.argptr[2] = unsafe.Pointer(&reducemaxdiff_args.arg_dst)
	 reducemaxdiff_args.argptr[3] = unsafe.Pointer(&reducemaxdiff_args.arg_initVal)
	 reducemaxdiff_args.argptr[4] = unsafe.Pointer(&reducemaxdiff_args.arg_n)
	 }

// Wrapper for reducemaxdiff CUDA kernel, asynchronous.
func k_reducemaxdiff_async ( src1 unsafe.Pointer, src2 unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("reducemaxdiff")
	}

	reducemaxdiff_args.Lock()
	defer reducemaxdiff_args.Unlock()

	if reducemaxdiff_code == 0{
		reducemaxdiff_code = fatbinLoad(reducemaxdiff_map, "reducemaxdiff")
	}

	 reducemaxdiff_args.arg_src1 = src1
	 reducemaxdiff_args.arg_src2 = src2
	 reducemaxdiff_args.arg_dst = dst
	 reducemaxdiff_args.arg_initVal = initVal
	 reducemaxdiff_args.arg_n = n
	

	args := reducemaxdiff_args.argptr[:]
	cu.LaunchKernel(reducemaxdiff_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("reducemaxdiff")
	}
}

// maps compute capability on PTX code for reducemaxdiff kernel.
var reducemaxdiff_map = map[int]string{ 0: "" ,
20: reducemaxdiff_ptx_20 ,
30: reducemaxdiff_ptx_30 ,
35: reducemaxdiff_ptx_35 ,
50: reducemaxdiff_ptx_50 ,
52: reducemaxdiff_ptx_52 ,
53: reducemaxdiff_ptx_53  }

// reducemaxdiff PTX code for various compute capabilities.
const(
  reducemaxdiff_ptx_20 = `
.version 4.3
.target sm_20
.address_size 64

	// .globl	reducemaxdiff

.visible .entry reducemaxdiff(
	.param .u64 reducemaxdiff_param_0,
	.param .u64 reducemaxdiff_param_1,
	.param .u64 reducemaxdiff_param_2,
	.param .f32 reducemaxdiff_param_3,
	.param .u32 reducemaxdiff_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<16>;
	// demoted variable
	.shared .align 4 .b8 reducemaxdiff$__cuda_local_var_42115_35_non_const_sdata[2048];

	ld.param.u64 	%rd5, [reducemaxdiff_param_0];
	ld.param.u64 	%rd6, [reducemaxdiff_param_1];
	ld.param.u64 	%rd4, [reducemaxdiff_param_2];
	ld.param.f32 	%f32, [reducemaxdiff_param_3];
	ld.param.u32 	%r9, [reducemaxdiff_param_4];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd5;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	add.s64 	%rd9, %rd1, %rd7;
	ld.global.f32 	%f5, [%rd9];
	ld.global.f32 	%f6, [%rd8];
	sub.f32 	%f7, %f6, %f5;
	abs.f32 	%f8, %f7;
	max.f32 	%f32, %f32, %f8;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	mul.wide.s32 	%rd10, %r2, 4;
	mov.u64 	%rd11, reducemaxdiff$__cuda_local_var_42115_35_non_const_sdata;
	add.s64 	%rd3, %rd11, %rd10;
	st.shared.f32 	[%rd3], %f32;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f9, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd12, %r12, 4;
	add.s64 	%rd14, %rd11, %rd12;
	ld.shared.f32 	%f10, [%rd14];
	max.f32 	%f11, %f9, %f10;
	st.shared.f32 	[%rd3], %f11;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f12, [%rd3];
	ld.volatile.shared.f32 	%f13, [%rd3+128];
	max.f32 	%f14, %f12, %f13;
	st.volatile.shared.f32 	[%rd3], %f14;
	ld.volatile.shared.f32 	%f15, [%rd3+64];
	ld.volatile.shared.f32 	%f16, [%rd3];
	max.f32 	%f17, %f16, %f15;
	st.volatile.shared.f32 	[%rd3], %f17;
	ld.volatile.shared.f32 	%f18, [%rd3+32];
	ld.volatile.shared.f32 	%f19, [%rd3];
	max.f32 	%f20, %f19, %f18;
	st.volatile.shared.f32 	[%rd3], %f20;
	ld.volatile.shared.f32 	%f21, [%rd3+16];
	ld.volatile.shared.f32 	%f22, [%rd3];
	max.f32 	%f23, %f22, %f21;
	st.volatile.shared.f32 	[%rd3], %f23;
	ld.volatile.shared.f32 	%f24, [%rd3+8];
	ld.volatile.shared.f32 	%f25, [%rd3];
	max.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%rd3], %f26;
	ld.volatile.shared.f32 	%f27, [%rd3+4];
	ld.volatile.shared.f32 	%f28, [%rd3];
	max.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%rd3], %f29;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f30, [reducemaxdiff$__cuda_local_var_42115_35_non_const_sdata];
	abs.f32 	%f31, %f30;
	mov.b32 	 %r13, %f31;
	cvta.to.global.u64 	%rd15, %rd4;
	atom.global.max.s32 	%r14, [%rd15], %r13;

BB0_10:
	ret;
}


`
   reducemaxdiff_ptx_30 = `
.version 4.3
.target sm_30
.address_size 64

	// .globl	reducemaxdiff

.visible .entry reducemaxdiff(
	.param .u64 reducemaxdiff_param_0,
	.param .u64 reducemaxdiff_param_1,
	.param .u64 reducemaxdiff_param_2,
	.param .f32 reducemaxdiff_param_3,
	.param .u32 reducemaxdiff_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<16>;
	// demoted variable
	.shared .align 4 .b8 reducemaxdiff$__cuda_local_var_42411_35_non_const_sdata[2048];

	ld.param.u64 	%rd5, [reducemaxdiff_param_0];
	ld.param.u64 	%rd6, [reducemaxdiff_param_1];
	ld.param.u64 	%rd4, [reducemaxdiff_param_2];
	ld.param.f32 	%f32, [reducemaxdiff_param_3];
	ld.param.u32 	%r9, [reducemaxdiff_param_4];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd5;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	add.s64 	%rd9, %rd1, %rd7;
	ld.global.f32 	%f5, [%rd9];
	ld.global.f32 	%f6, [%rd8];
	sub.f32 	%f7, %f6, %f5;
	abs.f32 	%f8, %f7;
	max.f32 	%f32, %f32, %f8;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	mul.wide.s32 	%rd10, %r2, 4;
	mov.u64 	%rd11, reducemaxdiff$__cuda_local_var_42411_35_non_const_sdata;
	add.s64 	%rd3, %rd11, %rd10;
	st.shared.f32 	[%rd3], %f32;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB0_5;

	ld.shared.f32 	%f9, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd12, %r12, 4;
	add.s64 	%rd14, %rd11, %rd12;
	ld.shared.f32 	%f10, [%rd14];
	max.f32 	%f11, %f9, %f10;
	st.shared.f32 	[%rd3], %f11;

BB0_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	ld.volatile.shared.f32 	%f12, [%rd3];
	ld.volatile.shared.f32 	%f13, [%rd3+128];
	max.f32 	%f14, %f12, %f13;
	st.volatile.shared.f32 	[%rd3], %f14;
	ld.volatile.shared.f32 	%f15, [%rd3+64];
	ld.volatile.shared.f32 	%f16, [%rd3];
	max.f32 	%f17, %f16, %f15;
	st.volatile.shared.f32 	[%rd3], %f17;
	ld.volatile.shared.f32 	%f18, [%rd3+32];
	ld.volatile.shared.f32 	%f19, [%rd3];
	max.f32 	%f20, %f19, %f18;
	st.volatile.shared.f32 	[%rd3], %f20;
	ld.volatile.shared.f32 	%f21, [%rd3+16];
	ld.volatile.shared.f32 	%f22, [%rd3];
	max.f32 	%f23, %f22, %f21;
	st.volatile.shared.f32 	[%rd3], %f23;
	ld.volatile.shared.f32 	%f24, [%rd3+8];
	ld.volatile.shared.f32 	%f25, [%rd3];
	max.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%rd3], %f26;
	ld.volatile.shared.f32 	%f27, [%rd3+4];
	ld.volatile.shared.f32 	%f28, [%rd3];
	max.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%rd3], %f29;

BB0_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	ld.shared.f32 	%f30, [reducemaxdiff$__cuda_local_var_42411_35_non_const_sdata];
	abs.f32 	%f31, %f30;
	mov.b32 	 %r13, %f31;
	cvta.to.global.u64 	%rd15, %rd4;
	atom.global.max.s32 	%r14, [%rd15], %r13;

BB0_10:
	ret;
}


`
   reducemaxdiff_ptx_35 = `
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

	// .globl	reducemaxdiff
.visible .entry reducemaxdiff(
	.param .u64 reducemaxdiff_param_0,
	.param .u64 reducemaxdiff_param_1,
	.param .u64 reducemaxdiff_param_2,
	.param .f32 reducemaxdiff_param_3,
	.param .u32 reducemaxdiff_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<16>;
	// demoted variable
	.shared .align 4 .b8 reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata[2048];

	ld.param.u64 	%rd5, [reducemaxdiff_param_0];
	ld.param.u64 	%rd6, [reducemaxdiff_param_1];
	ld.param.u64 	%rd4, [reducemaxdiff_param_2];
	ld.param.f32 	%f32, [reducemaxdiff_param_3];
	ld.param.u32 	%r9, [reducemaxdiff_param_4];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd5;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	add.s64 	%rd9, %rd1, %rd7;
	ld.global.nc.f32 	%f5, [%rd9];
	ld.global.nc.f32 	%f6, [%rd8];
	sub.f32 	%f7, %f6, %f5;
	abs.f32 	%f8, %f7;
	max.f32 	%f32, %f32, %f8;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd10, %r2, 4;
	mov.u64 	%rd11, reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata;
	add.s64 	%rd3, %rd11, %rd10;
	st.shared.f32 	[%rd3], %f32;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f9, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd12, %r12, 4;
	add.s64 	%rd14, %rd11, %rd12;
	ld.shared.f32 	%f10, [%rd14];
	max.f32 	%f11, %f9, %f10;
	st.shared.f32 	[%rd3], %f11;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f12, [%rd3];
	ld.volatile.shared.f32 	%f13, [%rd3+128];
	max.f32 	%f14, %f12, %f13;
	st.volatile.shared.f32 	[%rd3], %f14;
	ld.volatile.shared.f32 	%f15, [%rd3+64];
	ld.volatile.shared.f32 	%f16, [%rd3];
	max.f32 	%f17, %f16, %f15;
	st.volatile.shared.f32 	[%rd3], %f17;
	ld.volatile.shared.f32 	%f18, [%rd3+32];
	ld.volatile.shared.f32 	%f19, [%rd3];
	max.f32 	%f20, %f19, %f18;
	st.volatile.shared.f32 	[%rd3], %f20;
	ld.volatile.shared.f32 	%f21, [%rd3+16];
	ld.volatile.shared.f32 	%f22, [%rd3];
	max.f32 	%f23, %f22, %f21;
	st.volatile.shared.f32 	[%rd3], %f23;
	ld.volatile.shared.f32 	%f24, [%rd3+8];
	ld.volatile.shared.f32 	%f25, [%rd3];
	max.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%rd3], %f26;
	ld.volatile.shared.f32 	%f27, [%rd3+4];
	ld.volatile.shared.f32 	%f28, [%rd3];
	max.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%rd3], %f29;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f30, [reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata];
	abs.f32 	%f31, %f30;
	mov.b32 	 %r13, %f31;
	cvta.to.global.u64 	%rd15, %rd4;
	atom.global.max.s32 	%r14, [%rd15], %r13;

BB6_10:
	ret;
}


`
   reducemaxdiff_ptx_50 = `
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

	// .globl	reducemaxdiff
.visible .entry reducemaxdiff(
	.param .u64 reducemaxdiff_param_0,
	.param .u64 reducemaxdiff_param_1,
	.param .u64 reducemaxdiff_param_2,
	.param .f32 reducemaxdiff_param_3,
	.param .u32 reducemaxdiff_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<16>;
	// demoted variable
	.shared .align 4 .b8 reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata[2048];

	ld.param.u64 	%rd5, [reducemaxdiff_param_0];
	ld.param.u64 	%rd6, [reducemaxdiff_param_1];
	ld.param.u64 	%rd4, [reducemaxdiff_param_2];
	ld.param.f32 	%f32, [reducemaxdiff_param_3];
	ld.param.u32 	%r9, [reducemaxdiff_param_4];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd5;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	add.s64 	%rd9, %rd1, %rd7;
	ld.global.nc.f32 	%f5, [%rd9];
	ld.global.nc.f32 	%f6, [%rd8];
	sub.f32 	%f7, %f6, %f5;
	abs.f32 	%f8, %f7;
	max.f32 	%f32, %f32, %f8;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd10, %r2, 4;
	mov.u64 	%rd11, reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata;
	add.s64 	%rd3, %rd11, %rd10;
	st.shared.f32 	[%rd3], %f32;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f9, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd12, %r12, 4;
	add.s64 	%rd14, %rd11, %rd12;
	ld.shared.f32 	%f10, [%rd14];
	max.f32 	%f11, %f9, %f10;
	st.shared.f32 	[%rd3], %f11;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f12, [%rd3];
	ld.volatile.shared.f32 	%f13, [%rd3+128];
	max.f32 	%f14, %f12, %f13;
	st.volatile.shared.f32 	[%rd3], %f14;
	ld.volatile.shared.f32 	%f15, [%rd3+64];
	ld.volatile.shared.f32 	%f16, [%rd3];
	max.f32 	%f17, %f16, %f15;
	st.volatile.shared.f32 	[%rd3], %f17;
	ld.volatile.shared.f32 	%f18, [%rd3+32];
	ld.volatile.shared.f32 	%f19, [%rd3];
	max.f32 	%f20, %f19, %f18;
	st.volatile.shared.f32 	[%rd3], %f20;
	ld.volatile.shared.f32 	%f21, [%rd3+16];
	ld.volatile.shared.f32 	%f22, [%rd3];
	max.f32 	%f23, %f22, %f21;
	st.volatile.shared.f32 	[%rd3], %f23;
	ld.volatile.shared.f32 	%f24, [%rd3+8];
	ld.volatile.shared.f32 	%f25, [%rd3];
	max.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%rd3], %f26;
	ld.volatile.shared.f32 	%f27, [%rd3+4];
	ld.volatile.shared.f32 	%f28, [%rd3];
	max.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%rd3], %f29;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f30, [reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata];
	abs.f32 	%f31, %f30;
	mov.b32 	 %r13, %f31;
	cvta.to.global.u64 	%rd15, %rd4;
	atom.global.max.s32 	%r14, [%rd15], %r13;

BB6_10:
	ret;
}


`
   reducemaxdiff_ptx_52 = `
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

	// .globl	reducemaxdiff
.visible .entry reducemaxdiff(
	.param .u64 reducemaxdiff_param_0,
	.param .u64 reducemaxdiff_param_1,
	.param .u64 reducemaxdiff_param_2,
	.param .f32 reducemaxdiff_param_3,
	.param .u32 reducemaxdiff_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<16>;
	// demoted variable
	.shared .align 4 .b8 reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata[2048];

	ld.param.u64 	%rd5, [reducemaxdiff_param_0];
	ld.param.u64 	%rd6, [reducemaxdiff_param_1];
	ld.param.u64 	%rd4, [reducemaxdiff_param_2];
	ld.param.f32 	%f32, [reducemaxdiff_param_3];
	ld.param.u32 	%r9, [reducemaxdiff_param_4];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd5;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	add.s64 	%rd9, %rd1, %rd7;
	ld.global.nc.f32 	%f5, [%rd9];
	ld.global.nc.f32 	%f6, [%rd8];
	sub.f32 	%f7, %f6, %f5;
	abs.f32 	%f8, %f7;
	max.f32 	%f32, %f32, %f8;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd10, %r2, 4;
	mov.u64 	%rd11, reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata;
	add.s64 	%rd3, %rd11, %rd10;
	st.shared.f32 	[%rd3], %f32;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f9, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd12, %r12, 4;
	add.s64 	%rd14, %rd11, %rd12;
	ld.shared.f32 	%f10, [%rd14];
	max.f32 	%f11, %f9, %f10;
	st.shared.f32 	[%rd3], %f11;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f12, [%rd3];
	ld.volatile.shared.f32 	%f13, [%rd3+128];
	max.f32 	%f14, %f12, %f13;
	st.volatile.shared.f32 	[%rd3], %f14;
	ld.volatile.shared.f32 	%f15, [%rd3+64];
	ld.volatile.shared.f32 	%f16, [%rd3];
	max.f32 	%f17, %f16, %f15;
	st.volatile.shared.f32 	[%rd3], %f17;
	ld.volatile.shared.f32 	%f18, [%rd3+32];
	ld.volatile.shared.f32 	%f19, [%rd3];
	max.f32 	%f20, %f19, %f18;
	st.volatile.shared.f32 	[%rd3], %f20;
	ld.volatile.shared.f32 	%f21, [%rd3+16];
	ld.volatile.shared.f32 	%f22, [%rd3];
	max.f32 	%f23, %f22, %f21;
	st.volatile.shared.f32 	[%rd3], %f23;
	ld.volatile.shared.f32 	%f24, [%rd3+8];
	ld.volatile.shared.f32 	%f25, [%rd3];
	max.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%rd3], %f26;
	ld.volatile.shared.f32 	%f27, [%rd3+4];
	ld.volatile.shared.f32 	%f28, [%rd3];
	max.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%rd3], %f29;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f30, [reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata];
	abs.f32 	%f31, %f30;
	mov.b32 	 %r13, %f31;
	cvta.to.global.u64 	%rd15, %rd4;
	atom.global.max.s32 	%r14, [%rd15], %r13;

BB6_10:
	ret;
}


`
   reducemaxdiff_ptx_53 = `
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

	// .globl	reducemaxdiff
.visible .entry reducemaxdiff(
	.param .u64 reducemaxdiff_param_0,
	.param .u64 reducemaxdiff_param_1,
	.param .u64 reducemaxdiff_param_2,
	.param .f32 reducemaxdiff_param_3,
	.param .u32 reducemaxdiff_param_4
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<16>;
	// demoted variable
	.shared .align 4 .b8 reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata[2048];

	ld.param.u64 	%rd5, [reducemaxdiff_param_0];
	ld.param.u64 	%rd6, [reducemaxdiff_param_1];
	ld.param.u64 	%rd4, [reducemaxdiff_param_2];
	ld.param.f32 	%f32, [reducemaxdiff_param_3];
	ld.param.u32 	%r9, [reducemaxdiff_param_4];
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd5;
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r15, %r16, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r11, %r16;
	setp.ge.s32	%p1, %r15, %r9;
	@%p1 bra 	BB6_2;

BB6_1:
	mul.wide.s32 	%rd7, %r15, 4;
	add.s64 	%rd8, %rd2, %rd7;
	add.s64 	%rd9, %rd1, %rd7;
	ld.global.nc.f32 	%f5, [%rd9];
	ld.global.nc.f32 	%f6, [%rd8];
	sub.f32 	%f7, %f6, %f5;
	abs.f32 	%f8, %f7;
	max.f32 	%f32, %f32, %f8;
	add.s32 	%r15, %r15, %r4;
	setp.lt.s32	%p2, %r15, %r9;
	@%p2 bra 	BB6_1;

BB6_2:
	mul.wide.s32 	%rd10, %r2, 4;
	mov.u64 	%rd11, reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata;
	add.s64 	%rd3, %rd11, %rd10;
	st.shared.f32 	[%rd3], %f32;
	bar.sync 	0;
	setp.lt.u32	%p3, %r16, 66;
	@%p3 bra 	BB6_6;

BB6_3:
	mov.u32 	%r7, %r16;
	shr.u32 	%r16, %r7, 1;
	setp.ge.u32	%p4, %r2, %r16;
	@%p4 bra 	BB6_5;

	ld.shared.f32 	%f9, [%rd3];
	add.s32 	%r12, %r16, %r2;
	mul.wide.u32 	%rd12, %r12, 4;
	add.s64 	%rd14, %rd11, %rd12;
	ld.shared.f32 	%f10, [%rd14];
	max.f32 	%f11, %f9, %f10;
	st.shared.f32 	[%rd3], %f11;

BB6_5:
	bar.sync 	0;
	setp.gt.u32	%p5, %r7, 131;
	@%p5 bra 	BB6_3;

BB6_6:
	setp.gt.s32	%p6, %r2, 31;
	@%p6 bra 	BB6_8;

	ld.volatile.shared.f32 	%f12, [%rd3];
	ld.volatile.shared.f32 	%f13, [%rd3+128];
	max.f32 	%f14, %f12, %f13;
	st.volatile.shared.f32 	[%rd3], %f14;
	ld.volatile.shared.f32 	%f15, [%rd3+64];
	ld.volatile.shared.f32 	%f16, [%rd3];
	max.f32 	%f17, %f16, %f15;
	st.volatile.shared.f32 	[%rd3], %f17;
	ld.volatile.shared.f32 	%f18, [%rd3+32];
	ld.volatile.shared.f32 	%f19, [%rd3];
	max.f32 	%f20, %f19, %f18;
	st.volatile.shared.f32 	[%rd3], %f20;
	ld.volatile.shared.f32 	%f21, [%rd3+16];
	ld.volatile.shared.f32 	%f22, [%rd3];
	max.f32 	%f23, %f22, %f21;
	st.volatile.shared.f32 	[%rd3], %f23;
	ld.volatile.shared.f32 	%f24, [%rd3+8];
	ld.volatile.shared.f32 	%f25, [%rd3];
	max.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%rd3], %f26;
	ld.volatile.shared.f32 	%f27, [%rd3+4];
	ld.volatile.shared.f32 	%f28, [%rd3];
	max.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%rd3], %f29;

BB6_8:
	setp.ne.s32	%p7, %r2, 0;
	@%p7 bra 	BB6_10;

	ld.shared.f32 	%f30, [reducemaxdiff$__cuda_local_var_42668_35_non_const_sdata];
	abs.f32 	%f31, %f30;
	mov.b32 	 %r13, %f31;
	cvta.to.global.u64 	%rd15, %rd4;
	atom.global.max.s32 	%r14, [%rd15], %r13;

BB6_10:
	ret;
}


`
 )
