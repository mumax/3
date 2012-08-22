package ptx

const KERNMUL2D = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_000052fc_00000000-9_kernmul2d.cpp3.i (/tmp/ccBI#.WBktSa)
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
	.file	2	"/tmp/tmpxft_000052fc_00000000-8_kernmul2d.cudafe2.gpu"
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
	.file	14	"kernmul2d.cu"
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


	.entry kernmul2D (
		.param .u64 __cudaparm_kernmul2D_fftMx,
		.param .u64 __cudaparm_kernmul2D_fftMy,
		.param .u64 __cudaparm_kernmul2D_fftMz,
		.param .u64 __cudaparm_kernmul2D_fftKxx,
		.param .u64 __cudaparm_kernmul2D_fftKyy,
		.param .u64 __cudaparm_kernmul2D_fftKzz,
		.param .u64 __cudaparm_kernmul2D_fftKyz,
		.param .s32 __cudaparm_kernmul2D_N1,
		.param .s32 __cudaparm_kernmul2D_N2)
	{
	.reg .u16 %rh<6>;
	.reg .u32 %r<35>;
	.reg .u64 %rd<25>;
	.reg .f32 %f<42>;
	.reg .pred %p<4>;
	.loc	14	29	0
$LDWbegin_kernmul2D:
	mov.u16 	%rh1, %ctaid.y;
	mov.u16 	%rh2, %ntid.y;
	mul.wide.u16 	%r1, %rh1, %rh2;
	mov.u16 	%rh3, %ctaid.x;
	mov.u16 	%rh4, %ntid.x;
	mul.wide.u16 	%r2, %rh3, %rh4;
	cvt.u32.u16 	%r3, %tid.y;
	add.u32 	%r4, %r3, %r1;
	cvt.u32.u16 	%r5, %tid.x;
	add.u32 	%r6, %r5, %r2;
	ld.param.s32 	%r7, [__cudaparm_kernmul2D_N2];
	ld.param.s32 	%r8, [__cudaparm_kernmul2D_N1];
	set.le.u32.s32 	%r9, %r7, %r6;
	neg.s32 	%r10, %r9;
	set.le.u32.s32 	%r11, %r8, %r4;
	neg.s32 	%r12, %r11;
	or.b32 	%r13, %r10, %r12;
	mov.u32 	%r14, 0;
	setp.eq.s32 	%p1, %r13, %r14;
	@%p1 bra 	$Lt_0_2306;
	bra.uni 	$LBB6_kernmul2D;
$Lt_0_2306:
	ld.param.s32 	%r7, [__cudaparm_kernmul2D_N2];
	.loc	14	40	0
	mul.lo.s32 	%r15, %r7, %r4;
	add.s32 	%r16, %r6, %r15;
	cvt.s64.s32 	%rd1, %r16;
	mul.wide.s32 	%rd2, %r16, 4;
	ld.param.u64 	%rd3, [__cudaparm_kernmul2D_fftKxx];
	add.u64 	%rd4, %rd3, %rd2;
	ld.global.f32 	%f1, [%rd4+0];
	.loc	14	41	0
	ld.param.u64 	%rd5, [__cudaparm_kernmul2D_fftKyy];
	add.u64 	%rd6, %rd5, %rd2;
	ld.global.f32 	%f2, [%rd6+0];
	.loc	14	42	0
	ld.param.u64 	%rd7, [__cudaparm_kernmul2D_fftKzz];
	add.u64 	%rd8, %rd7, %rd2;
	ld.global.f32 	%f3, [%rd8+0];
	.loc	14	43	0
	ld.param.u64 	%rd9, [__cudaparm_kernmul2D_fftKyz];
	add.u64 	%rd10, %rd9, %rd2;
	ld.global.f32 	%f4, [%rd10+0];
	.loc	14	49	0
	mul.lo.s32 	%r17, %r16, 2;
	cvt.s64.s32 	%rd11, %r17;
	mul.wide.s32 	%rd12, %r17, 4;
	ld.param.u64 	%rd13, [__cudaparm_kernmul2D_fftMx];
	add.u64 	%rd14, %rd12, %rd13;
	ld.global.f32 	%f5, [%rd14+4];
	.loc	14	50	0
	ld.param.u64 	%rd15, [__cudaparm_kernmul2D_fftMy];
	add.u64 	%rd16, %rd12, %rd15;
	ld.global.f32 	%f6, [%rd16+0];
	.loc	14	51	0
	ld.global.f32 	%f7, [%rd16+4];
	.loc	14	52	0
	ld.param.u64 	%rd17, [__cudaparm_kernmul2D_fftMz];
	add.u64 	%rd18, %rd12, %rd17;
	ld.global.f32 	%f8, [%rd18+0];
	.loc	14	53	0
	ld.global.f32 	%f9, [%rd18+4];
	.loc	14	55	0
	ld.global.f32 	%f10, [%rd14+0];
	mul.f32 	%f11, %f10, %f1;
	st.global.f32 	[%rd14+0], %f11;
	.loc	14	56	0
	mul.f32 	%f12, %f1, %f5;
	st.global.f32 	[%rd14+4], %f12;
	.loc	14	57	0
	mul.f32 	%f13, %f4, %f8;
	mad.f32 	%f14, %f2, %f6, %f13;
	st.global.f32 	[%rd16+0], %f14;
	.loc	14	58	0
	mul.f32 	%f15, %f4, %f9;
	mad.f32 	%f16, %f2, %f7, %f15;
	st.global.f32 	[%rd16+4], %f16;
	.loc	14	59	0
	mul.f32 	%f17, %f4, %f6;
	mad.f32 	%f18, %f3, %f8, %f17;
	st.global.f32 	[%rd18+0], %f18;
	.loc	14	60	0
	mul.f32 	%f19, %f4, %f7;
	mad.f32 	%f20, %f3, %f9, %f19;
	st.global.f32 	[%rd18+4], %f20;
	mov.s32 	%r18, 0;
	set.gt.u32.s32 	%r19, %r4, %r18;
	neg.s32 	%r20, %r19;
	.loc	14	29	0
	ld.param.s32 	%r8, [__cudaparm_kernmul2D_N1];
	.loc	14	60	0
	shr.s32 	%r21, %r8, 31;
	mov.s32 	%r22, 1;
	and.b32 	%r23, %r21, %r22;
	add.s32 	%r24, %r23, %r8;
	shr.s32 	%r25, %r24, 1;
	set.lt.u32.s32 	%r26, %r4, %r25;
	neg.s32 	%r27, %r26;
	and.b32 	%r28, %r20, %r27;
	mov.u32 	%r29, 0;
	setp.eq.s32 	%p2, %r28, %r29;
	@%p2 bra 	$LBB6_kernmul2D;
	.loc	14	29	0
	ld.param.s32 	%r8, [__cudaparm_kernmul2D_N1];
	.loc	14	69	0
	sub.s32 	%r30, %r8, %r4;
	.loc	14	29	0
	ld.param.s32 	%r7, [__cudaparm_kernmul2D_N2];
	.loc	14	69	0
	mul.lo.s32 	%r31, %r30, %r7;
	add.s32 	%r32, %r31, %r6;
	mul.lo.s32 	%r33, %r32, 2;
	cvt.s64.s32 	%rd19, %r33;
	mul.wide.s32 	%rd20, %r33, 4;
	.loc	14	49	0
	ld.param.u64 	%rd13, [__cudaparm_kernmul2D_fftMx];
	.loc	14	69	0
	add.u64 	%rd21, %rd20, %rd13;
	ld.global.f32 	%f21, [%rd21+4];
	.loc	14	50	0
	ld.param.u64 	%rd15, [__cudaparm_kernmul2D_fftMy];
	.loc	14	70	0
	add.u64 	%rd22, %rd20, %rd15;
	ld.global.f32 	%f22, [%rd22+0];
	.loc	14	71	0
	ld.global.f32 	%f23, [%rd22+4];
	.loc	14	52	0
	ld.param.u64 	%rd17, [__cudaparm_kernmul2D_fftMz];
	.loc	14	72	0
	add.u64 	%rd23, %rd20, %rd17;
	ld.global.f32 	%f24, [%rd23+0];
	.loc	14	73	0
	ld.global.f32 	%f25, [%rd23+4];
	.loc	14	75	0
	ld.global.f32 	%f26, [%rd21+0];
	mul.f32 	%f27, %f26, %f1;
	st.global.f32 	[%rd21+0], %f27;
	.loc	14	76	0
	mul.f32 	%f28, %f1, %f21;
	st.global.f32 	[%rd21+4], %f28;
	.loc	14	77	0
	mul.f32 	%f29, %f4, %f24;
	mul.f32 	%f30, %f2, %f22;
	sub.f32 	%f31, %f30, %f29;
	st.global.f32 	[%rd22+0], %f31;
	.loc	14	78	0
	mul.f32 	%f32, %f4, %f25;
	mul.f32 	%f33, %f2, %f23;
	sub.f32 	%f34, %f33, %f32;
	st.global.f32 	[%rd22+4], %f34;
	.loc	14	79	0
	mul.f32 	%f35, %f4, %f22;
	mul.f32 	%f36, %f3, %f24;
	sub.f32 	%f37, %f36, %f35;
	st.global.f32 	[%rd23+0], %f37;
	.loc	14	80	0
	mul.f32 	%f38, %f4, %f23;
	mul.f32 	%f39, %f3, %f25;
	sub.f32 	%f40, %f39, %f38;
	st.global.f32 	[%rd23+4], %f40;
$LBB6_kernmul2D:
	.loc	14	82	0
	exit;
$LDWend_kernmul2D:
	} // kernmul2D

`
