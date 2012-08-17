package ptx

const COPYPAD = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_00007298_00000000-9_copypad.cpp3.i (/tmp/ccBI#.fdx7R9)
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
	.file	2	"/tmp/tmpxft_00007298_00000000-8_copypad.cudafe2.gpu"
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
	.reg .u32 %r<72>;
	.reg .u64 %rd<10>;
	.reg .f32 %f<3>;
	.reg .pred %p<6>;
	.loc	14	14	0
$LDWbegin_copypad:
$LDWbeginblock_234_1:
	.loc	14	16	0
	cvt.u32.u16 	%r1, %tid.y;
	cvt.u32.u16 	%r2, %ctaid.y;
	cvt.u32.u16 	%r3, %ntid.y;
	mul.lo.u32 	%r4, %r2, %r3;
	add.u32 	%r5, %r1, %r4;
	mov.s32 	%r6, %r5;
	.loc	14	17	0
	cvt.u32.u16 	%r7, %tid.x;
	cvt.u32.u16 	%r8, %ctaid.x;
	cvt.u32.u16 	%r9, %ntid.x;
	mul.lo.u32 	%r10, %r8, %r9;
	add.u32 	%r11, %r7, %r10;
	mov.s32 	%r12, %r11;
	.loc	14	19	0
	ld.param.s32 	%r13, [__cudaparm_copypad_S1];
	mov.s32 	%r14, %r6;
	set.le.u32.s32 	%r15, %r13, %r14;
	neg.s32 	%r16, %r15;
	ld.param.s32 	%r17, [__cudaparm_copypad_S2];
	mov.s32 	%r18, %r12;
	set.le.u32.s32 	%r19, %r17, %r18;
	neg.s32 	%r20, %r19;
	or.b32 	%r21, %r16, %r20;
	mov.u32 	%r22, 0;
	setp.eq.s32 	%p1, %r21, %r22;
	@%p1 bra 	$L_0_4098;
	bra.uni 	$LBB12_copypad;
$L_0_4098:
	.loc	14	22	0
	ld.param.s32 	%r23, [__cudaparm_copypad_D1];
	mov.s32 	%r24, %r6;
	set.le.u32.s32 	%r25, %r23, %r24;
	neg.s32 	%r26, %r25;
	ld.param.s32 	%r27, [__cudaparm_copypad_D2];
	mov.s32 	%r28, %r12;
	set.le.u32.s32 	%r29, %r27, %r28;
	neg.s32 	%r30, %r29;
	or.b32 	%r31, %r26, %r30;
	mov.u32 	%r32, 0;
	setp.eq.s32 	%p2, %r31, %r32;
	@%p2 bra 	$L_0_4354;
	bra.uni 	$LBB12_copypad;
$L_0_4354:
	.loc	14	26	0
	ld.param.s32 	%r33, [__cudaparm_copypad_o1];
	mov.s32 	%r34, %r6;
	add.s32 	%r35, %r33, %r34;
	mov.s32 	%r36, %r35;
	.loc	14	27	0
	ld.param.s32 	%r37, [__cudaparm_copypad_o2];
	mov.s32 	%r38, %r12;
	add.s32 	%r39, %r37, %r38;
	mov.s32 	%r40, %r39;
$LDWbeginblock_234_3:
	.loc	14	29	0
	mov.s32 	%r41, 0;
	mov.s32 	%r42, %r41;
	ld.param.s32 	%r43, [__cudaparm_copypad_S0];
	mov.s32 	%r44, %r42;
	setp.le.s32 	%p3, %r43, %r44;
	@%p3 bra 	$L_0_4866;
$L_0_4610:
$LDWbeginblock_234_5:
	.loc	14	30	0
	ld.param.s32 	%r45, [__cudaparm_copypad_o0];
	mov.s32 	%r46, %r42;
	add.s32 	%r47, %r45, %r46;
	mov.s32 	%r48, %r47;
	.loc	14	32	0
	ld.param.u64 	%rd1, [__cudaparm_copypad_src];
	mov.s32 	%r49, %r12;
	ld.param.s32 	%r50, [__cudaparm_copypad_S2];
	mov.s32 	%r51, %r6;
	ld.param.s32 	%r52, [__cudaparm_copypad_S1];
	mov.s32 	%r53, %r42;
	mul.lo.s32 	%r54, %r52, %r53;
	add.s32 	%r55, %r51, %r54;
	mul.lo.s32 	%r56, %r50, %r55;
	add.s32 	%r57, %r49, %r56;
	cvt.s64.s32 	%rd2, %r57;
	mul.wide.s32 	%rd3, %r57, 4;
	add.u64 	%rd4, %rd1, %rd3;
	ld.global.f32 	%f1, [%rd4+0];
	ld.param.u64 	%rd5, [__cudaparm_copypad_dst];
	mov.s32 	%r58, %r40;
	ld.param.s32 	%r59, [__cudaparm_copypad_D2];
	mov.s32 	%r60, %r36;
	ld.param.s32 	%r61, [__cudaparm_copypad_D1];
	mov.s32 	%r62, %r48;
	mul.lo.s32 	%r63, %r61, %r62;
	add.s32 	%r64, %r60, %r63;
	mul.lo.s32 	%r65, %r59, %r64;
	add.s32 	%r66, %r58, %r65;
	cvt.s64.s32 	%rd6, %r66;
	mul.wide.s32 	%rd7, %r66, 4;
	add.u64 	%rd8, %rd5, %rd7;
	st.global.f32 	[%rd8+0], %f1;
$LDWendblock_234_5:
	.loc	14	29	0
	mov.s32 	%r67, %r42;
	add.s32 	%r68, %r67, 1;
	mov.s32 	%r42, %r68;
$Lt_0_1794:
	ld.param.s32 	%r69, [__cudaparm_copypad_S0];
	mov.s32 	%r70, %r42;
	setp.gt.s32 	%p4, %r69, %r70;
	@%p4 bra 	$L_0_4610;
$L_0_4866:
$LDWendblock_234_3:
$LDWendblock_234_1:
$LBB12_copypad:
	.loc	14	34	0
	exit;
$LDWend_copypad:
	} // copypad

 	@@DWARF .section .debug_info, "",@progbits
	@@DWARF .byte	0x34, 0x04, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_abbrev
	@@DWARF .4byte	0x742f0108, 0x742f706d, 0x6678706d, 0x30305f74
	@@DWARF .4byte	0x32373030, 0x305f3839, 0x30303030, 0x2d303030
	@@DWARF .4byte	0x6f635f39, 0x61707970, 0x70632e64, 0x692e3370
	@@DWARF .4byte	0x6f682f00, 0x612f656d, 0x2f656e72, 0x732f6f67
	@@DWARF .4byte	0x2f736372, 0x626d696e, 0x632d656c, 0x2f656275
	@@DWARF .4byte	0x2f757067, 0x00787470, 0x6e65706f, 0x34206363
	@@DWARF .byte	0x2e, 0x31, 0x00, 0x04, 0x00
	@@DWARF .4byte	.debug_line
	@@DWARF .4byte	0x736e7502, 0x656e6769, 0x6e692064, 0x04070074
	@@DWARF .4byte	0x75bb0b03, 0x33746e69, 0x00b90c00, 0x0b040000
	@@DWARF .4byte	0x730078bc, 0x02000000, 0x04010023, 0x0079bc0b
	@@DWARF .4byte	0x00000073, 0x01042302, 0x7abc0b04, 0x00007300
	@@DWARF .4byte	0x08230200, 0x0b050001, 0x69750167, 0x0033746e
	@@DWARF .4byte	0x00000083, 0x0000b906, 0x8a0b0700, 0x6d696401
	@@DWARF .4byte	0x050c0033, 0x08000001, 0x78018b0b, 0x00007300
	@@DWARF .4byte	0x00230200, 0x8b0b0801, 0x73007901, 0x02000000
	@@DWARF .4byte	0x08010423, 0x7a018b0b, 0x00007300, 0x08230200
	@@DWARF .4byte	0x0b050001, 0x69640193, 0xcc00336d, 0x06000000
	@@DWARF .4byte	0x00000105, 0x746e6902, 0x06040500, 0x00000117
	@@DWARF .4byte	0x6f6c6602, 0x04007461, 0x01230904, 0x05080000
	@@DWARF .4byte	0x756f6402, 0x00656c62, 0x6c020804, 0x20676e6f
	@@DWARF .4byte	0x676e6f6c, 0x736e7520, 0x656e6769, 0x6e692064
	@@DWARF .4byte	0x08070074, 0x6e6f6c02, 0x6f6c2067, 0x6920676e
	@@DWARF .4byte	0x0500746e, 0x3e070a08, 0x61647563, 0x6e756f52
	@@DWARF .4byte	0x646f4d64, 0xda040065, 0x0b000001, 0x75633f07
	@@DWARF .4byte	0x6f526164, 0x4e646e75, 0x65726165, 0x00007473
	@@DWARF .4byte	0x0b000000, 0x75634007, 0x6f526164, 0x5a646e75
	@@DWARF .4byte	0x006f7265, 0x00000001, 0x6341070b, 0x52616475
	@@DWARF .4byte	0x646e756f, 0x49736f50, 0x0200666e, 0x0b000000
	@@DWARF .4byte	0x75634307, 0x6f526164, 0x4d646e75, 0x6e496e69
	@@DWARF .4byte	0x00030066, 0x02000000, 0x676e6f6c, 0x746e6920
	@@DWARF .4byte	0x02080500, 0x72616863, 0x06010600, 0x000001e6
	@@DWARF .4byte	0x0001ee09, 0x03050800, 0x6c66fd0b, 0x3274616f
	@@DWARF .4byte	0x02240800, 0x0b040000, 0x230078fd, 0x02000001
	@@DWARF .4byte	0x04010023, 0x0079fd0b, 0x00000123, 0x01042302
	@@DWARF .4byte	0x730b0500, 0x6f6c6601, 0x00327461, 0x000001fa
	@@DWARF .4byte	0x0000730c, 0x02400100, 0x050d0000, 0x14100e00
	@@DWARF .4byte	0x635f5f0d, 0x72616475, 0x32695f74, 0x5f69706f
	@@DWARF .4byte	0x02330066, 0x17090000, 0x08000001, 0x00730c05
	@@DWARF .4byte	0x6d010000, 0x0d000002, 0x230f0006, 0x06000001
	@@DWARF .4byte	0x00000123, 0x00013309, 0x02050800, 0x676e6f6c
	@@DWARF .4byte	0x736e7520, 0x656e6769, 0x6e692064, 0x08070074
	@@DWARF .byte	0x10, 0x0e, 0x0e, 0x63, 0x6f, 0x70, 0x79, 0x70
	@@DWARF .byte	0x61, 0x64, 0x00, 0x01, 0x01, 0x00
	@@DWARF .quad	$LDWbegin_copypad
	@@DWARF .quad	$LDWend_copypad
	@@DWARF .byte	0x11, 0x0e, 0x0e, 0x64, 0x73, 0x74, 0x00, 0x2c
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_dst
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x44, 0x30, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_D0
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x44, 0x31, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_D1
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x44, 0x32, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_D2
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x73, 0x72, 0x63, 0x00
	@@DWARF .byte	0x2c, 0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_src
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x53, 0x30, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_S0
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x53, 0x31, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_S1
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x53, 0x32, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_S2
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x6f, 0x30, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_o0
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x6f, 0x31, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_o1
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x0e, 0x6f, 0x32, 0x00, 0x17
	@@DWARF .byte	0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_copypad_o2
	@@DWARF .byte	0x07, 0x12
	@@DWARF .quad	$LDWbeginblock_234_1
	@@DWARF .quad	$LDWendblock_234_1
	@@DWARF .4byte	0x6a100e13, 0x00011700, 0xb6900500, 0x020195e4
	@@DWARF .4byte	0x6b110e13, 0x00011700, 0xb2900600, 0x02abc8e2
	@@DWARF .4byte	0x1a0e1302, 0x0117004a, 0x90060000, 0xabc8e6b6
	@@DWARF .4byte	0x0e130202, 0x17004b1b, 0x06000001, 0xc8e8b090
	@@DWARF .byte	0xab, 0x02, 0x02, 0x12
	@@DWARF .quad	$LDWbeginblock_234_3
	@@DWARF .quad	$LDWendblock_234_3
	@@DWARF .4byte	0x691d0e13, 0x00011700, 0xb2900600, 0x02abc8e8
	@@DWARF .byte	0x02, 0x12
	@@DWARF .quad	$LDWbeginblock_234_5
	@@DWARF .quad	$LDWendblock_234_5
	@@DWARF .4byte	0x491e0e13, 0x00011700, 0xb8900600, 0x02abc8e8
	@@DWARF .byte	0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

 	@@DWARF .section .debug_pubnames, "",@progbits
	@@DWARF .byte	0x1a, 0x00, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_info
	@@DWARF .4byte	0x00000438, 0x00000293, 0x79706f63, 0x00646170
	@@DWARF .byte	0x00, 0x00, 0x00, 0x00

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
