package ptx

const KERNMUL2D = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_000002e5_00000000-9_kernmul2d.cpp3.i (/tmp/ccBI#.L7cCxj)
	//-----------------------------------------------------------

	//-----------------------------------------------------------
	// Options:
	//-----------------------------------------------------------
	//  Target:ptx, ISA:sm_10, Endian:little, Pointer Size:64
	//  -O0	(Optimization level)
	//  -g2	(Debug level)
	//  -m2	(Report advisories)
	//-----------------------------------------------------------

	.file	1	"<command-line>"
	.file	2	"/tmp/tmpxft_000002e5_00000000-8_kernmul2d.cudafe2.gpu"
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
	.reg .u32 %r<45>;
	.reg .u64 %rd<114>;
	.reg .f32 %f<88>;
	.reg .pred %p<4>;
	.loc	14	29	0
$LDWbegin_kernmul2D:
$LDWbeginblock_234_1:
	.loc	14	31	0
	cvt.u32.u16 	%r1, %tid.y;
	cvt.u32.u16 	%r2, %ctaid.y;
	cvt.u32.u16 	%r3, %ntid.y;
	mul.lo.u32 	%r4, %r2, %r3;
	add.u32 	%r5, %r1, %r4;
	mov.s32 	%r6, %r5;
	.loc	14	32	0
	cvt.u32.u16 	%r7, %tid.x;
	cvt.u32.u16 	%r8, %ctaid.x;
	cvt.u32.u16 	%r9, %ntid.x;
	mul.lo.u32 	%r10, %r8, %r9;
	add.u32 	%r11, %r7, %r10;
	mov.s32 	%r12, %r11;
	.loc	14	34	0
	ld.param.s32 	%r13, [__cudaparm_kernmul2D_N1];
	mov.s32 	%r14, %r6;
	set.le.u32.s32 	%r15, %r13, %r14;
	neg.s32 	%r16, %r15;
	ld.param.s32 	%r17, [__cudaparm_kernmul2D_N2];
	mov.s32 	%r18, %r12;
	set.le.u32.s32 	%r19, %r17, %r18;
	neg.s32 	%r20, %r19;
	or.b32 	%r21, %r16, %r20;
	mov.u32 	%r22, 0;
	setp.eq.s32 	%p1, %r21, %r22;
	@%p1 bra 	$L_0_2050;
	bra.uni 	$LBB7_kernmul2D;
$L_0_2050:
	.loc	14	38	0
	mov.s32 	%r23, %r12;
	ld.param.s32 	%r24, [__cudaparm_kernmul2D_N2];
	mov.s32 	%r25, %r6;
	mul.lo.s32 	%r26, %r24, %r25;
	add.s32 	%r27, %r23, %r26;
	mov.s32 	%r28, %r27;
	.loc	14	40	0
	ld.param.u64 	%rd1, [__cudaparm_kernmul2D_fftKxx];
	cvt.s64.s32 	%rd2, %r28;
	mul.wide.s32 	%rd3, %r28, 4;
	add.u64 	%rd4, %rd1, %rd3;
	ld.global.f32 	%f1, [%rd4+0];
	mov.f32 	%f2, %f1;
	.loc	14	41	0
	ld.param.u64 	%rd5, [__cudaparm_kernmul2D_fftKyy];
	cvt.s64.s32 	%rd6, %r28;
	mul.wide.s32 	%rd7, %r28, 4;
	add.u64 	%rd8, %rd5, %rd7;
	ld.global.f32 	%f3, [%rd8+0];
	mov.f32 	%f4, %f3;
	.loc	14	42	0
	ld.param.u64 	%rd9, [__cudaparm_kernmul2D_fftKzz];
	cvt.s64.s32 	%rd10, %r28;
	mul.wide.s32 	%rd11, %r28, 4;
	add.u64 	%rd12, %rd9, %rd11;
	ld.global.f32 	%f5, [%rd12+0];
	mov.f32 	%f6, %f5;
	.loc	14	43	0
	ld.param.u64 	%rd13, [__cudaparm_kernmul2D_fftKyz];
	cvt.s64.s32 	%rd14, %r28;
	mul.wide.s32 	%rd15, %r28, 4;
	add.u64 	%rd16, %rd13, %rd15;
	ld.global.f32 	%f7, [%rd16+0];
	mov.f32 	%f8, %f7;
	.loc	14	46	0
	mov.s32 	%r29, %r28;
	mul.lo.s32 	%r30, %r29, 2;
	mov.s32 	%r31, %r30;
	.loc	14	48	0
	ld.param.u64 	%rd17, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd18, %r31;
	mul.wide.s32 	%rd19, %r31, 4;
	add.u64 	%rd20, %rd17, %rd19;
	ld.global.f32 	%f9, [%rd20+0];
	mov.f32 	%f10, %f9;
	.loc	14	49	0
	ld.param.u64 	%rd21, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd22, %r31;
	mul.wide.s32 	%rd23, %r31, 4;
	add.u64 	%rd24, %rd21, %rd23;
	ld.global.f32 	%f11, [%rd24+4];
	mov.f32 	%f12, %f11;
	.loc	14	50	0
	ld.param.u64 	%rd25, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd26, %r31;
	mul.wide.s32 	%rd27, %r31, 4;
	add.u64 	%rd28, %rd25, %rd27;
	ld.global.f32 	%f13, [%rd28+0];
	mov.f32 	%f14, %f13;
	.loc	14	51	0
	ld.param.u64 	%rd29, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd30, %r31;
	mul.wide.s32 	%rd31, %r31, 4;
	add.u64 	%rd32, %rd29, %rd31;
	ld.global.f32 	%f15, [%rd32+4];
	mov.f32 	%f16, %f15;
	.loc	14	52	0
	ld.param.u64 	%rd33, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd34, %r31;
	mul.wide.s32 	%rd35, %r31, 4;
	add.u64 	%rd36, %rd33, %rd35;
	ld.global.f32 	%f17, [%rd36+0];
	mov.f32 	%f18, %f17;
	.loc	14	53	0
	ld.param.u64 	%rd37, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd38, %r31;
	mul.wide.s32 	%rd39, %r31, 4;
	add.u64 	%rd40, %rd37, %rd39;
	ld.global.f32 	%f19, [%rd40+4];
	mov.f32 	%f20, %f19;
	.loc	14	55	0
	mov.f32 	%f21, %f2;
	mov.f32 	%f22, %f10;
	mul.f32 	%f23, %f21, %f22;
	ld.param.u64 	%rd41, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd42, %r31;
	mul.wide.s32 	%rd43, %r31, 4;
	add.u64 	%rd44, %rd41, %rd43;
	st.global.f32 	[%rd44+0], %f23;
	.loc	14	56	0
	mov.f32 	%f24, %f2;
	mov.f32 	%f25, %f12;
	mul.f32 	%f26, %f24, %f25;
	ld.param.u64 	%rd45, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd46, %r31;
	mul.wide.s32 	%rd47, %r31, 4;
	add.u64 	%rd48, %rd45, %rd47;
	st.global.f32 	[%rd48+4], %f26;
	.loc	14	57	0
	mov.f32 	%f27, %f8;
	mov.f32 	%f28, %f18;
	mul.f32 	%f29, %f27, %f28;
	mov.f32 	%f30, %f4;
	mov.f32 	%f31, %f14;
	mad.f32 	%f32, %f30, %f31, %f29;
	ld.param.u64 	%rd49, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd50, %r31;
	mul.wide.s32 	%rd51, %r31, 4;
	add.u64 	%rd52, %rd49, %rd51;
	st.global.f32 	[%rd52+0], %f32;
	.loc	14	58	0
	mov.f32 	%f33, %f8;
	mov.f32 	%f34, %f20;
	mul.f32 	%f35, %f33, %f34;
	mov.f32 	%f36, %f4;
	mov.f32 	%f37, %f16;
	mad.f32 	%f38, %f36, %f37, %f35;
	ld.param.u64 	%rd53, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd54, %r31;
	mul.wide.s32 	%rd55, %r31, 4;
	add.u64 	%rd56, %rd53, %rd55;
	st.global.f32 	[%rd56+4], %f38;
	.loc	14	59	0
	mov.f32 	%f39, %f8;
	mov.f32 	%f40, %f14;
	mul.f32 	%f41, %f39, %f40;
	mov.f32 	%f42, %f6;
	mov.f32 	%f43, %f18;
	mad.f32 	%f44, %f42, %f43, %f41;
	ld.param.u64 	%rd57, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd58, %r31;
	mul.wide.s32 	%rd59, %r31, 4;
	add.u64 	%rd60, %rd57, %rd59;
	st.global.f32 	[%rd60+0], %f44;
	.loc	14	60	0
	mov.f32 	%f45, %f8;
	mov.f32 	%f46, %f16;
	mul.f32 	%f47, %f45, %f46;
	mov.f32 	%f48, %f6;
	mov.f32 	%f49, %f20;
	mad.f32 	%f50, %f48, %f49, %f47;
	ld.param.u64 	%rd61, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd62, %r31;
	mul.wide.s32 	%rd63, %r31, 4;
	add.u64 	%rd64, %rd61, %rd63;
	st.global.f32 	[%rd64+4], %f50;
	.loc	14	64	0
	mov.s32 	%r32, %r6;
	mov.u32 	%r33, 0;
	setp.le.s32 	%p2, %r32, %r33;
	@%p2 bra 	$L_0_2306;
	.loc	14	65	0
	ld.param.s32 	%r34, [__cudaparm_kernmul2D_N1];
	mov.s32 	%r35, %r6;
	sub.s32 	%r36, %r34, %r35;
	mov.s32 	%r6, %r36;
	.loc	14	66	0
	mov.s32 	%r37, %r12;
	ld.param.s32 	%r38, [__cudaparm_kernmul2D_N2];
	mov.s32 	%r39, %r6;
	mul.lo.s32 	%r40, %r38, %r39;
	add.s32 	%r41, %r37, %r40;
	mov.s32 	%r28, %r41;
	.loc	14	67	0
	mov.s32 	%r42, %r28;
	mul.lo.s32 	%r43, %r42, 2;
	mov.s32 	%r31, %r43;
	.loc	14	69	0
	ld.param.u64 	%rd65, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd66, %r31;
	mul.wide.s32 	%rd67, %r31, 4;
	add.u64 	%rd68, %rd65, %rd67;
	ld.global.f32 	%f51, [%rd68+0];
	mov.f32 	%f10, %f51;
	.loc	14	70	0
	ld.param.u64 	%rd69, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd70, %r31;
	mul.wide.s32 	%rd71, %r31, 4;
	add.u64 	%rd72, %rd69, %rd71;
	ld.global.f32 	%f52, [%rd72+4];
	mov.f32 	%f12, %f52;
	.loc	14	71	0
	ld.param.u64 	%rd73, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd74, %r31;
	mul.wide.s32 	%rd75, %r31, 4;
	add.u64 	%rd76, %rd73, %rd75;
	ld.global.f32 	%f53, [%rd76+0];
	mov.f32 	%f14, %f53;
	.loc	14	72	0
	ld.param.u64 	%rd77, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd78, %r31;
	mul.wide.s32 	%rd79, %r31, 4;
	add.u64 	%rd80, %rd77, %rd79;
	ld.global.f32 	%f54, [%rd80+4];
	mov.f32 	%f16, %f54;
	.loc	14	73	0
	ld.param.u64 	%rd81, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd82, %r31;
	mul.wide.s32 	%rd83, %r31, 4;
	add.u64 	%rd84, %rd81, %rd83;
	ld.global.f32 	%f55, [%rd84+0];
	mov.f32 	%f18, %f55;
	.loc	14	74	0
	ld.param.u64 	%rd85, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd86, %r31;
	mul.wide.s32 	%rd87, %r31, 4;
	add.u64 	%rd88, %rd85, %rd87;
	ld.global.f32 	%f56, [%rd88+4];
	mov.f32 	%f20, %f56;
	.loc	14	76	0
	mov.f32 	%f57, %f2;
	mov.f32 	%f58, %f10;
	mul.f32 	%f59, %f57, %f58;
	ld.param.u64 	%rd89, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd90, %r31;
	mul.wide.s32 	%rd91, %r31, 4;
	add.u64 	%rd92, %rd89, %rd91;
	st.global.f32 	[%rd92+0], %f59;
	.loc	14	77	0
	mov.f32 	%f60, %f2;
	mov.f32 	%f61, %f12;
	mul.f32 	%f62, %f60, %f61;
	ld.param.u64 	%rd93, [__cudaparm_kernmul2D_fftMx];
	cvt.s64.s32 	%rd94, %r31;
	mul.wide.s32 	%rd95, %r31, 4;
	add.u64 	%rd96, %rd93, %rd95;
	st.global.f32 	[%rd96+4], %f62;
	.loc	14	78	0
	mov.f32 	%f63, %f8;
	mov.f32 	%f64, %f18;
	mul.f32 	%f65, %f63, %f64;
	mov.f32 	%f66, %f4;
	mov.f32 	%f67, %f14;
	mad.f32 	%f68, %f66, %f67, %f65;
	ld.param.u64 	%rd97, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd98, %r31;
	mul.wide.s32 	%rd99, %r31, 4;
	add.u64 	%rd100, %rd97, %rd99;
	st.global.f32 	[%rd100+0], %f68;
	.loc	14	79	0
	mov.f32 	%f69, %f8;
	mov.f32 	%f70, %f20;
	mul.f32 	%f71, %f69, %f70;
	mov.f32 	%f72, %f4;
	mov.f32 	%f73, %f16;
	mad.f32 	%f74, %f72, %f73, %f71;
	ld.param.u64 	%rd101, [__cudaparm_kernmul2D_fftMy];
	cvt.s64.s32 	%rd102, %r31;
	mul.wide.s32 	%rd103, %r31, 4;
	add.u64 	%rd104, %rd101, %rd103;
	st.global.f32 	[%rd104+4], %f74;
	.loc	14	80	0
	mov.f32 	%f75, %f8;
	mov.f32 	%f76, %f14;
	mul.f32 	%f77, %f75, %f76;
	mov.f32 	%f78, %f6;
	mov.f32 	%f79, %f18;
	mad.f32 	%f80, %f78, %f79, %f77;
	ld.param.u64 	%rd105, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd106, %r31;
	mul.wide.s32 	%rd107, %r31, 4;
	add.u64 	%rd108, %rd105, %rd107;
	st.global.f32 	[%rd108+0], %f80;
	.loc	14	81	0
	mov.f32 	%f81, %f8;
	mov.f32 	%f82, %f16;
	mul.f32 	%f83, %f81, %f82;
	mov.f32 	%f84, %f6;
	mov.f32 	%f85, %f20;
	mad.f32 	%f86, %f84, %f85, %f83;
	ld.param.u64 	%rd109, [__cudaparm_kernmul2D_fftMz];
	cvt.s64.s32 	%rd110, %r31;
	mul.wide.s32 	%rd111, %r31, 4;
	add.u64 	%rd112, %rd109, %rd111;
	st.global.f32 	[%rd112+4], %f86;
$L_0_2306:
$LDWendblock_234_1:
$LBB7_kernmul2D:
	.loc	14	83	0
	exit;
$LDWend_kernmul2D:
	} // kernmul2D

 	@@DWARF .section .debug_info, "",@progbits
	@@DWARF .byte	0x9a, 0x04, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_abbrev
	@@DWARF .4byte	0x742f0108, 0x742f706d, 0x6678706d, 0x30305f74
	@@DWARF .4byte	0x32303030, 0x305f3565, 0x30303030, 0x2d303030
	@@DWARF .4byte	0x656b5f39, 0x756d6e72, 0x2e64326c, 0x33707063
	@@DWARF .4byte	0x2f00692e, 0x656d6f68, 0x6e72612f, 0x6f672f65
	@@DWARF .4byte	0x6372732f, 0x6d696e2f, 0x2d656c62, 0x65627563
	@@DWARF .4byte	0x7874702f, 0x65706f00, 0x2063636e, 0x00312e34
	@@DWARF .byte	0x04, 0x00
	@@DWARF .4byte	.debug_line
	@@DWARF .4byte	0x736e7502, 0x656e6769, 0x6e692064, 0x04070074
	@@DWARF .4byte	0x75bb0b03, 0x33746e69, 0x00b60c00, 0x0b040000
	@@DWARF .4byte	0x700078bc, 0x02000000, 0x04010023, 0x0079bc0b
	@@DWARF .4byte	0x00000070, 0x01042302, 0x7abc0b04, 0x00007000
	@@DWARF .4byte	0x08230200, 0x0b050001, 0x69750167, 0x0033746e
	@@DWARF .4byte	0x00000080, 0x0000b606, 0x8a0b0700, 0x6d696401
	@@DWARF .4byte	0x020c0033, 0x08000001, 0x78018b0b, 0x00007000
	@@DWARF .4byte	0x00230200, 0x8b0b0801, 0x70007901, 0x02000000
	@@DWARF .4byte	0x08010423, 0x7a018b0b, 0x00007000, 0x08230200
	@@DWARF .4byte	0x0b050001, 0x69640193, 0xc900336d, 0x06000000
	@@DWARF .4byte	0x00000102, 0x746e6902, 0x06040500, 0x00000114
	@@DWARF .4byte	0x6f6c6602, 0x04007461, 0x01200904, 0x05080000
	@@DWARF .4byte	0x756f6402, 0x00656c62, 0x6c020804, 0x20676e6f
	@@DWARF .4byte	0x676e6f6c, 0x736e7520, 0x656e6769, 0x6e692064
	@@DWARF .4byte	0x08070074, 0x6e6f6c02, 0x6f6c2067, 0x6920676e
	@@DWARF .4byte	0x0500746e, 0x3e070a08, 0x61647563, 0x6e756f52
	@@DWARF .4byte	0x646f4d64, 0xd7040065, 0x0b000001, 0x75633f07
	@@DWARF .4byte	0x6f526164, 0x4e646e75, 0x65726165, 0x00007473
	@@DWARF .4byte	0x0b000000, 0x75634007, 0x6f526164, 0x5a646e75
	@@DWARF .4byte	0x006f7265, 0x00000001, 0x6341070b, 0x52616475
	@@DWARF .4byte	0x646e756f, 0x49736f50, 0x0200666e, 0x0b000000
	@@DWARF .4byte	0x75634307, 0x6f526164, 0x4d646e75, 0x6e496e69
	@@DWARF .4byte	0x00030066, 0x02000000, 0x676e6f6c, 0x746e6920
	@@DWARF .4byte	0x02080500, 0x72616863, 0x06010600, 0x000001e3
	@@DWARF .4byte	0x0001eb09, 0x03050800, 0x6c66fd0b, 0x3274616f
	@@DWARF .4byte	0x02210800, 0x0b040000, 0x200078fd, 0x02000001
	@@DWARF .4byte	0x04010023, 0x0079fd0b, 0x00000120, 0x01042302
	@@DWARF .4byte	0x730b0500, 0x6f6c6601, 0x00327461, 0x000001f7
	@@DWARF .4byte	0x0000700c, 0x023d0100, 0x050d0000, 0x14100e00
	@@DWARF .4byte	0x635f5f0d, 0x72616475, 0x32695f74, 0x5f69706f
	@@DWARF .4byte	0x02300066, 0x14090000, 0x08000001, 0x00700c05
	@@DWARF .4byte	0x6a010000, 0x0d000002, 0x200f0006, 0x06000001
	@@DWARF .4byte	0x00000120, 0x00013009, 0x02050800, 0x676e6f6c
	@@DWARF .4byte	0x736e7520, 0x656e6769, 0x6e692064, 0x08070074
	@@DWARF .4byte	0x6b1d0e10, 0x6d6e7265, 0x44326c75, 0x00010100
	@@DWARF .quad	$LDWbegin_kernmul2D
	@@DWARF .quad	$LDWend_kernmul2D
	@@DWARF .byte	0x11, 0x0e, 0x1d, 0x66, 0x66, 0x74, 0x4d, 0x78
	@@DWARF .byte	0x00, 0x29, 0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_kernmul2D_fftMx
	@@DWARF .4byte	0x1d0e1107, 0x4d746666, 0x01290079, 0x03090000
	@@DWARF .quad	__cudaparm_kernmul2D_fftMy
	@@DWARF .4byte	0x1d0e1107, 0x4d746666, 0x0129007a, 0x03090000
	@@DWARF .quad	__cudaparm_kernmul2D_fftMz
	@@DWARF .4byte	0x1d0e1107, 0x4b746666, 0x29007878, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul2D_fftKxx
	@@DWARF .4byte	0x1d0e1107, 0x4b746666, 0x29007979, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul2D_fftKyy
	@@DWARF .4byte	0x1d0e1107, 0x4b746666, 0x29007a7a, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul2D_fftKzz
	@@DWARF .4byte	0x1d0e1107, 0x4b746666, 0x29007a79, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul2D_fftKyz
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x1d, 0x4e, 0x31, 0x00, 0x14
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_kernmul2D_N1
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x1d, 0x4e, 0x32, 0x00, 0x14
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_kernmul2D_N2
	@@DWARF .byte	0x07, 0x12
	@@DWARF .quad	$LDWbeginblock_234_1
	@@DWARF .quad	$LDWendblock_234_1
	@@DWARF .4byte	0x6a1f0e13, 0x00011400, 0xb6900500, 0x020195e4
	@@DWARF .4byte	0x6b200e13, 0x00011400, 0xb2900600, 0x02abc8e2
	@@DWARF .4byte	0x260e1302, 0x01140049, 0x90060000, 0xabc8e4b8
	@@DWARF .4byte	0x0e130202, 0x78784b28, 0x00012000, 0xb2900500
	@@DWARF .4byte	0x020195cc, 0x4b290e13, 0x20007979, 0x05000001
	@@DWARF .4byte	0x95ccb490, 0x0e130201, 0x7a7a4b2a, 0x00012000
	@@DWARF .4byte	0xb6900500, 0x020195cc, 0x4b2b0e13, 0x20007a79
	@@DWARF .4byte	0x05000001, 0x95ccb890, 0x0e130201, 0x1400652e
	@@DWARF .4byte	0x06000001, 0xc8e6b190, 0x130202ab, 0x6572300e
	@@DWARF .4byte	0x2000784d, 0x06000001, 0x98e2b090, 0x130202ab
	@@DWARF .4byte	0x6d69310e, 0x2000784d, 0x06000001, 0x98e2b290
	@@DWARF .4byte	0x130202ab, 0x6572320e, 0x2000794d, 0x06000001
	@@DWARF .4byte	0x98e2b490, 0x130202ab, 0x6d69330e, 0x2000794d
	@@DWARF .4byte	0x06000001, 0x98e2b690, 0x130202ab, 0x6572340e
	@@DWARF .4byte	0x20007a4d, 0x06000001, 0x98e2b890, 0x130202ab
	@@DWARF .4byte	0x6d69350e, 0x20007a4d, 0x06000001, 0x98e4b090
	@@DWARF .byte	0xab, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00

 	@@DWARF .section .debug_pubnames, "",@progbits
	@@DWARF .byte	0x1c, 0x00, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_info
	@@DWARF .4byte	0x0000049e, 0x00000290, 0x6e72656b, 0x326c756d
	@@DWARF .byte	0x44, 0x00, 0x00, 0x00, 0x00, 0x00

 	@@DWARF .section .debug_abbrev, "",@progbits
	@@DWARF .4byte	0x03011101, 0x25081b08, 0x420b1308, 0x0006100b
	@@DWARF .4byte	0x00240200, 0x0b3e0803, 0x00000b0b, 0x3a011303
	@@DWARF .4byte	0x030b3b0b, 0x010b0b08, 0x04000013, 0x0b3a000d
	@@DWARF .4byte	0x08030b3b, 0x0a381349, 0x00000b32, 0x3a001605
	@@DWARF .4byte	0x03053b0b, 0x00134908, 0x00260600, 0x00001349
	@@DWARF .4byte	0x3a011307, 0x03053b0b, 0x010b0b08, 0x08000013
	@@DWARF .4byte	0x0b3a000d, 0x0803053b, 0x0a381349, 0x00000b32
	@@DWARF .4byte	0x49000f09, 0x330b0b13, 0x0a00000b, 0x0b3a0104
	@@DWARF .4byte	0x08030b3b, 0x13010b0b, 0x280b0000, 0x3b0b3a00
	@@DWARF .4byte	0x1c08030b, 0x0c000006, 0x13490101, 0x13010c3c
	@@DWARF .4byte	0x210d0000, 0x000b2f00, 0x00340e00, 0x053b0b3a
	@@DWARF .4byte	0x13490803, 0x350f0000, 0x00134900, 0x012e1000
	@@DWARF .4byte	0x0b3b0b3a, 0x0c3f0803, 0x0a400c27, 0x01120111
	@@DWARF .4byte	0x05110000, 0x3b0b3a00, 0x4908030b, 0x330a0213
	@@DWARF .4byte	0x1200000b, 0x0111010b, 0x00000112, 0x3a003413
	@@DWARF .byte	0x0b, 0x3b, 0x0b, 0x03, 0x08, 0x49, 0x13, 0x02
	@@DWARF .byte	0x0a, 0x33, 0x0b, 0x00, 0x00, 0x00, 0x00

`
