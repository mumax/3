package ptx

const COPYPAD = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_000052d2_00000000-9_copypad.cpp3.i (/tmp/ccBI#.JJlNza)
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
	.file	2	"/tmp/tmpxft_000052d2_00000000-8_copypad.cudafe2.gpu"
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
	.reg .u32 %r<42>;
	.reg .u64 %rd<14>;
	.reg .f32 %f<3>;
	.reg .pred %p<6>;
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
	@%p1 bra 	$Lt_0_3330;
	bra.uni 	$LBB10_copypad;
$Lt_0_3330:
	.loc	14	20	0
	ld.param.s32 	%r15, [__cudaparm_copypad_D2];
	ld.param.s32 	%r16, [__cudaparm_copypad_D1];
	set.le.u32.s32 	%r17, %r15, %r6;
	neg.s32 	%r18, %r17;
	set.le.u32.s32 	%r19, %r16, %r4;
	neg.s32 	%r20, %r19;
	or.b32 	%r21, %r18, %r20;
	mov.u32 	%r22, 0;
	setp.eq.s32 	%p2, %r21, %r22;
	@%p2 bra 	$Lt_0_5378;
	bra.uni 	$LBB10_copypad;
$Lt_0_5378:
	.loc	14	29	0
	ld.param.s32 	%r23, [__cudaparm_copypad_S0];
	mov.u32 	%r24, 0;
	setp.le.s32 	%p3, %r23, %r24;
	@%p3 bra 	$LBB10_copypad;
	ld.param.s32 	%r23, [__cudaparm_copypad_S0];
	mov.s32 	%r25, %r23;
	ld.param.s32 	%r26, [__cudaparm_copypad_o0];
	.loc	14	20	0
	ld.param.s32 	%r16, [__cudaparm_copypad_D1];
	.loc	14	29	0
	mul.lo.s32 	%r27, %r26, %r16;
	ld.param.s32 	%r28, [__cudaparm_copypad_o1];
	add.s32 	%r29, %r28, %r4;
	add.s32 	%r30, %r29, %r27;
	.loc	14	14	0
	ld.param.s32 	%r7, [__cudaparm_copypad_S2];
	ld.param.s32 	%r8, [__cudaparm_copypad_S1];
	.loc	14	29	0
	mul.lo.s32 	%r31, %r7, %r8;
	cvt.s64.s32 	%rd1, %r31;
	mul.wide.s32 	%rd2, %r31, 4;
	ld.param.u64 	%rd3, [__cudaparm_copypad_src];
	mul.lo.s32 	%r32, %r7, %r4;
	add.s32 	%r33, %r6, %r32;
	cvt.s64.s32 	%rd4, %r33;
	mul.wide.s32 	%rd5, %r33, 4;
	add.u64 	%rd6, %rd3, %rd5;
	.loc	14	20	0
	ld.param.s32 	%r15, [__cudaparm_copypad_D2];
	.loc	14	29	0
	mul.lo.s32 	%r34, %r15, %r16;
	cvt.s64.s32 	%rd7, %r34;
	mul.wide.s32 	%rd8, %r34, 4;
	ld.param.u64 	%rd9, [__cudaparm_copypad_dst];
	ld.param.s32 	%r35, [__cudaparm_copypad_o2];
	add.s32 	%r36, %r35, %r6;
	mul.lo.s32 	%r37, %r30, %r15;
	add.s32 	%r38, %r36, %r37;
	cvt.s64.s32 	%rd10, %r38;
	mul.wide.s32 	%rd11, %r38, 4;
	add.u64 	%rd12, %rd9, %rd11;
	mov.s32 	%r39, 0;
	mov.s32 	%r40, %r25;
$Lt_0_4866:
 //<loop> Loop body line 29, nesting depth: 1, estimated iterations: unknown
	.loc	14	32	0
	ld.global.f32 	%f1, [%rd6+0];
	st.global.f32 	[%rd12+0], %f1;
	add.s32 	%r39, %r39, 1;
	add.u64 	%rd12, %rd8, %rd12;
	add.u64 	%rd6, %rd2, %rd6;
	.loc	14	29	0
	ld.param.s32 	%r23, [__cudaparm_copypad_S0];
	.loc	14	32	0
	setp.ne.s32 	%p4, %r23, %r39;
	@%p4 bra 	$Lt_0_4866;
$LBB10_copypad:
	.loc	14	34	0
	exit;
$LDWend_copypad:
	} // copypad

`
