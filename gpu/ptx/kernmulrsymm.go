package ptx

const KERNMULRSYMM = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_00005352_00000000-9_kernmulrsymm.cpp3.i (/tmp/ccBI#.G8tQNb)
	//-----------------------------------------------------------

	//-----------------------------------------------------------
	// Options:
	//-----------------------------------------------------------
	//  Target:ptx, ISA:sm_10, Endian:little, Pointer Size:64
	//  -O3	(Optimization level)
	//  -g0	(Debug level)
	//  -m2	(Report advisories)
	//-----------------------------------------------------------

	.file	1	"<command-line>"
	.file	2	"/tmp/tmpxft_00005352_00000000-8_kernmulrsymm.cudafe2.gpu"
	.file	3	"/usr/lib/gcc/x86_64-linux-gnu/4.6/include/stddef.h"
	.file	4	"/usr/local/cuda/bin/../include/crt/device_runtime.h"
	.file	5	"/usr/local/cuda/bin/../include/host_defines.h"
	.file	6	"/usr/local/cuda/bin/../include/builtin_types.h"
	.file	7	"/usr/local/cuda/bin/../include/device_types.h"
	.file	8	"/usr/local/cuda/bin/../include/driver_types.h"
	.file	9	"/usr/local/cuda/bin/../include/surface_types.h"
	.file	10	"/usr/local/cuda/bin/../include/texture_types.h"
	.file	11	"/usr/local/cuda/bin/../include/vector_types.h"
	.file	12	"/usr/local/cuda/bin/../include/device_launch_parameters.h"
	.file	13	"/usr/local/cuda/bin/../include/crt/storage_class.h"
	.file	14	"kernmulrsymm.cu"
	.file	15	"/usr/local/cuda/bin/../include/common_functions.h"
	.file	16	"/usr/local/cuda/bin/../include/math_functions.h"
	.file	17	"/usr/local/cuda/bin/../include/math_constants.h"
	.file	18	"/usr/local/cuda/bin/../include/device_functions.h"
	.file	19	"/usr/local/cuda/bin/../include/sm_11_atomic_functions.h"
	.file	20	"/usr/local/cuda/bin/../include/sm_12_atomic_functions.h"
	.file	21	"/usr/local/cuda/bin/../include/sm_13_double_functions.h"
	.file	22	"/usr/local/cuda/bin/../include/sm_20_atomic_functions.h"
	.file	23	"/usr/local/cuda/bin/../include/sm_20_intrinsics.h"
	.file	24	"/usr/local/cuda/bin/../include/sm_30_intrinsics.h"
	.file	25	"/usr/local/cuda/bin/../include/surface_functions.h"
	.file	26	"/usr/local/cuda/bin/../include/texture_fetch_functions.h"
	.file	27	"/usr/local/cuda/bin/../include/math_functions_dbl_ptx1.h"


	.entry kernmulRSymm (
		.param .u64 __cudaparm_kernmulRSymm_fftMx,
		.param .u64 __cudaparm_kernmulRSymm_fftMy,
		.param .u64 __cudaparm_kernmulRSymm_fftMz,
		.param .u64 __cudaparm_kernmulRSymm_fftKxx,
		.param .u64 __cudaparm_kernmulRSymm_fftKyy,
		.param .u64 __cudaparm_kernmulRSymm_fftKzz,
		.param .u64 __cudaparm_kernmulRSymm_fftKyz,
		.param .u64 __cudaparm_kernmulRSymm_fftKxz,
		.param .u64 __cudaparm_kernmulRSymm_fftKxy,
		.param .s32 __cudaparm_kernmulRSymm_N)
	{
	.reg .u16 %rh<4>;
	.reg .u32 %r<11>;
	.reg .u64 %rd<24>;
	.reg .f32 %f<32>;
	.reg .pred %p<3>;
	.loc	14	10	0
$LDWbegin_kernmulRSymm:
	mov.u16 	%rh1, %nctaid.x;
	mov.u16 	%rh2, %ctaid.y;
	mul.wide.u16 	%r1, %rh1, %rh2;
	cvt.u32.u16 	%r2, %ctaid.x;
	add.u32 	%r3, %r2, %r1;
	cvt.u32.u16 	%r4, %ntid.x;
	mul.lo.u32 	%r5, %r4, %r3;
	cvt.u32.u16 	%r6, %tid.x;
	add.u32 	%r7, %r6, %r5;
	ld.param.s32 	%r8, [__cudaparm_kernmulRSymm_N];
	setp.le.s32 	%p1, %r8, %r7;
	@%p1 bra 	$Lt_0_1026;
	.loc	14	16	0
	mul.lo.s32 	%r9, %r7, 2;
	cvt.s64.s32 	%rd1, %r9;
	mul.wide.s32 	%rd2, %r9, 4;
	ld.param.u64 	%rd3, [__cudaparm_kernmulRSymm_fftMx];
	add.u64 	%rd4, %rd3, %rd2;
	ld.global.f32 	%f1, [%rd4+0];
	.loc	14	17	0
	ld.global.f32 	%f2, [%rd4+4];
	.loc	14	19	0
	ld.param.u64 	%rd5, [__cudaparm_kernmulRSymm_fftMy];
	add.u64 	%rd6, %rd5, %rd2;
	ld.global.f32 	%f3, [%rd6+0];
	.loc	14	20	0
	ld.global.f32 	%f4, [%rd6+4];
	.loc	14	22	0
	ld.param.u64 	%rd7, [__cudaparm_kernmulRSymm_fftMz];
	add.u64 	%rd8, %rd7, %rd2;
	ld.global.f32 	%f5, [%rd8+0];
	.loc	14	23	0
	ld.global.f32 	%f6, [%rd8+4];
	.loc	14	25	0
	cvt.s64.s32 	%rd9, %r7;
	mul.wide.s32 	%rd10, %r7, 4;
	ld.param.u64 	%rd11, [__cudaparm_kernmulRSymm_fftKxx];
	add.u64 	%rd12, %rd11, %rd10;
	ld.global.f32 	%f7, [%rd12+0];
	.loc	14	26	0
	ld.param.u64 	%rd13, [__cudaparm_kernmulRSymm_fftKyy];
	add.u64 	%rd14, %rd13, %rd10;
	ld.global.f32 	%f8, [%rd14+0];
	.loc	14	27	0
	ld.param.u64 	%rd15, [__cudaparm_kernmulRSymm_fftKzz];
	add.u64 	%rd16, %rd15, %rd10;
	ld.global.f32 	%f9, [%rd16+0];
	.loc	14	29	0
	ld.param.u64 	%rd17, [__cudaparm_kernmulRSymm_fftKyz];
	add.u64 	%rd18, %rd17, %rd10;
	ld.global.f32 	%f10, [%rd18+0];
	.loc	14	30	0
	ld.param.u64 	%rd19, [__cudaparm_kernmulRSymm_fftKxz];
	add.u64 	%rd20, %rd19, %rd10;
	ld.global.f32 	%f11, [%rd20+0];
	.loc	14	31	0
	ld.param.u64 	%rd21, [__cudaparm_kernmulRSymm_fftKxy];
	add.u64 	%rd22, %rd21, %rd10;
	ld.global.f32 	%f12, [%rd22+0];
	.loc	14	33	0
	mul.f32 	%f13, %f3, %f12;
	mad.f32 	%f14, %f1, %f7, %f13;
	mad.f32 	%f15, %f5, %f11, %f14;
	st.global.f32 	[%rd4+0], %f15;
	.loc	14	34	0
	mul.f32 	%f16, %f4, %f12;
	mad.f32 	%f17, %f2, %f7, %f16;
	mad.f32 	%f18, %f6, %f11, %f17;
	st.global.f32 	[%rd4+4], %f18;
	.loc	14	36	0
	mul.f32 	%f19, %f3, %f8;
	mad.f32 	%f20, %f1, %f12, %f19;
	mad.f32 	%f21, %f5, %f10, %f20;
	st.global.f32 	[%rd6+0], %f21;
	.loc	14	37	0
	mul.f32 	%f22, %f4, %f8;
	mad.f32 	%f23, %f2, %f12, %f22;
	mad.f32 	%f24, %f6, %f10, %f23;
	st.global.f32 	[%rd6+4], %f24;
	.loc	14	39	0
	mul.f32 	%f25, %f3, %f10;
	mad.f32 	%f26, %f1, %f11, %f25;
	mad.f32 	%f27, %f5, %f9, %f26;
	st.global.f32 	[%rd8+0], %f27;
	.loc	14	40	0
	mul.f32 	%f28, %f4, %f10;
	mad.f32 	%f29, %f2, %f11, %f28;
	mad.f32 	%f30, %f6, %f9, %f29;
	st.global.f32 	[%rd8+4], %f30;
$Lt_0_1026:
	.loc	14	42	0
	exit;
$LDWend_kernmulRSymm:
	} // kernmulRSymm

`
