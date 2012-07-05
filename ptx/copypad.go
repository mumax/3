package ptx

const COPYPAD = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_00004d7d_00000000-9_copypad.cpp3.i (/tmp/ccBI#.67LCzP)
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
	.file	2	"/tmp/tmpxft_00004d7d_00000000-8_copypad.cudafe2.gpu"
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
	.file	14	"copypad.cu"
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


	.entry copypad (
		.param .u64 __cudaparm_copypad_dst,
		.param .s32 __cudaparm_copypad_D0,
		.param .s32 __cudaparm_copypad_D1,
		.param .s32 __cudaparm_copypad_D2,
		.param .u64 __cudaparm_copypad_src,
		.param .s32 __cudaparm_copypad_S0,
		.param .s32 __cudaparm_copypad_S1,
		.param .s32 __cudaparm_copypad_S2,
		.param .s32 __cudaparm_copypad_o0,
		.param .s32 __cudaparm_copypad_o1,
		.param .s32 __cudaparm_copypad_o2)
	{
	.reg .u16 %rh<6>;
	.reg .u32 %r<36>;
	.reg .u64 %rd<14>;
	.reg .f32 %f<3>;
	.reg .pred %p<5>;
	.loc	14	14	0
$LDWbegin_copypad:
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
	ld.param.s32 	%r7, [__cudaparm_copypad_S2];
	ld.param.s32 	%r8, [__cudaparm_copypad_S1];
	set.le.u32.s32 	%r9, %r7, %r6;
	neg.s32 	%r10, %r9;
	set.le.u32.s32 	%r11, %r8, %r4;
	neg.s32 	%r12, %r11;
	or.b32 	%r13, %r10, %r12;
	mov.u32 	%r14, 0;
	setp.eq.s32 	%p1, %r13, %r14;
	@%p1 bra 	$Lt_0_3842;
	bra.uni 	$LBB8_copypad;
$Lt_0_3842:
	.loc	14	26	0
	ld.param.s32 	%r15, [__cudaparm_copypad_S0];
	mov.u32 	%r16, 0;
	setp.le.s32 	%p2, %r15, %r16;
	@%p2 bra 	$LBB8_copypad;
	ld.param.s32 	%r15, [__cudaparm_copypad_S0];
	mov.s32 	%r17, %r15;
	ld.param.s32 	%r18, [__cudaparm_copypad_D1];
	ld.param.s32 	%r19, [__cudaparm_copypad_o0];
	mul.lo.s32 	%r20, %r19, %r18;
	ld.param.s32 	%r21, [__cudaparm_copypad_o1];
	add.s32 	%r22, %r21, %r4;
	add.s32 	%r23, %r22, %r20;
	ld.param.s32 	%r24, [__cudaparm_copypad_D2];
	.loc	14	14	0
	ld.param.s32 	%r7, [__cudaparm_copypad_S2];
	ld.param.s32 	%r8, [__cudaparm_copypad_S1];
	.loc	14	26	0
	mul.lo.s32 	%r25, %r7, %r8;
	cvt.s64.s32 	%rd1, %r25;
	mul.wide.s32 	%rd2, %r25, 4;
	ld.param.u64 	%rd3, [__cudaparm_copypad_src];
	mul.lo.s32 	%r26, %r7, %r4;
	add.s32 	%r27, %r6, %r26;
	cvt.s64.s32 	%rd4, %r27;
	mul.wide.s32 	%rd5, %r27, 4;
	add.u64 	%rd6, %rd3, %rd5;
	mul.lo.s32 	%r28, %r24, %r18;
	cvt.s64.s32 	%rd7, %r28;
	mul.wide.s32 	%rd8, %r28, 4;
	ld.param.u64 	%rd9, [__cudaparm_copypad_dst];
	ld.param.s32 	%r29, [__cudaparm_copypad_o2];
	add.s32 	%r30, %r29, %r6;
	mul.lo.s32 	%r31, %r23, %r24;
	add.s32 	%r32, %r30, %r31;
	cvt.s64.s32 	%rd10, %r32;
	mul.wide.s32 	%rd11, %r32, 4;
	add.u64 	%rd12, %rd9, %rd11;
	mov.s32 	%r33, 0;
	mov.s32 	%r34, %r17;
$Lt_0_3330:
 //<loop> Loop body line 26, nesting depth: 1, estimated iterations: unknown
	.loc	14	29	0
	ld.global.f32 	%f1, [%rd6+0];
	st.global.f32 	[%rd12+0], %f1;
	add.s32 	%r33, %r33, 1;
	add.u64 	%rd12, %rd8, %rd12;
	add.u64 	%rd6, %rd2, %rd6;
	.loc	14	26	0
	ld.param.s32 	%r15, [__cudaparm_copypad_S0];
	.loc	14	29	0
	setp.ne.s32 	%p3, %r15, %r33;
	@%p3 bra 	$Lt_0_3330;
$LBB8_copypad:
	.loc	14	31	0
	exit;
$LDWend_copypad:
	} // copypad

`
