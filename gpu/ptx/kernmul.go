package ptx

const KERNMUL = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_0000595e_00000000-9_kernmul.cpp3.i (/tmp/ccBI#.TEt4Ml)
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
	.file	2	"/tmp/tmpxft_0000595e_00000000-8_kernmul.cudafe2.gpu"
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
	.file	14	"kernmul.cu"
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


	.entry kernmul (
		.param .u64 __cudaparm_kernmul_fftMx,
		.param .u64 __cudaparm_kernmul_fftMy,
		.param .u64 __cudaparm_kernmul_fftMz,
		.param .u64 __cudaparm_kernmul_fftKxx,
		.param .u64 __cudaparm_kernmul_fftKyy,
		.param .u64 __cudaparm_kernmul_fftKzz,
		.param .u64 __cudaparm_kernmul_fftKyz,
		.param .u64 __cudaparm_kernmul_fftKxz,
		.param .u64 __cudaparm_kernmul_fftKxy,
		.param .s32 __cudaparm_kernmul_N)
	{
	.reg .u32 %r<17>;
	.reg .u64 %rd<74>;
	.reg .f32 %f<80>;
	.reg .pred %p<3>;
	.loc	14	9	0
$LDWbegin_kernmul:
$LDWbeginblock_234_1:
	.loc	14	11	0
	cvt.u32.u16 	%r1, %tid.x;
	cvt.u32.u16 	%r2, %ntid.x;
	cvt.u32.u16 	%r3, %ctaid.x;
	cvt.u32.u16 	%r4, %nctaid.x;
	cvt.u32.u16 	%r5, %ctaid.y;
	mul.lo.u32 	%r6, %r4, %r5;
	add.u32 	%r7, %r3, %r6;
	mul.lo.u32 	%r8, %r2, %r7;
	add.u32 	%r9, %r1, %r8;
	mov.s32 	%r10, %r9;
	.loc	14	12	0
	mov.s32 	%r11, %r10;
	mul.lo.s32 	%r12, %r11, 2;
	mov.s32 	%r13, %r12;
	.loc	14	14	0
	ld.param.s32 	%r14, [__cudaparm_kernmul_N];
	mov.s32 	%r15, %r10;
	setp.le.s32 	%p1, %r14, %r15;
	@%p1 bra 	$L_0_1794;
$LDWbeginblock_234_3:
	.loc	14	15	0
	ld.param.u64 	%rd1, [__cudaparm_kernmul_fftMx];
	cvt.s64.s32 	%rd2, %r13;
	mul.wide.s32 	%rd3, %r13, 4;
	add.u64 	%rd4, %rd1, %rd3;
	ld.global.f32 	%f1, [%rd4+0];
	mov.f32 	%f2, %f1;
	.loc	14	16	0
	ld.param.u64 	%rd5, [__cudaparm_kernmul_fftMx];
	cvt.s64.s32 	%rd6, %r13;
	mul.wide.s32 	%rd7, %r13, 4;
	add.u64 	%rd8, %rd5, %rd7;
	ld.global.f32 	%f3, [%rd8+4];
	mov.f32 	%f4, %f3;
	.loc	14	18	0
	ld.param.u64 	%rd9, [__cudaparm_kernmul_fftMy];
	cvt.s64.s32 	%rd10, %r13;
	mul.wide.s32 	%rd11, %r13, 4;
	add.u64 	%rd12, %rd9, %rd11;
	ld.global.f32 	%f5, [%rd12+0];
	mov.f32 	%f6, %f5;
	.loc	14	19	0
	ld.param.u64 	%rd13, [__cudaparm_kernmul_fftMy];
	cvt.s64.s32 	%rd14, %r13;
	mul.wide.s32 	%rd15, %r13, 4;
	add.u64 	%rd16, %rd13, %rd15;
	ld.global.f32 	%f7, [%rd16+4];
	mov.f32 	%f8, %f7;
	.loc	14	21	0
	ld.param.u64 	%rd17, [__cudaparm_kernmul_fftMz];
	cvt.s64.s32 	%rd18, %r13;
	mul.wide.s32 	%rd19, %r13, 4;
	add.u64 	%rd20, %rd17, %rd19;
	ld.global.f32 	%f9, [%rd20+0];
	mov.f32 	%f10, %f9;
	.loc	14	22	0
	ld.param.u64 	%rd21, [__cudaparm_kernmul_fftMz];
	cvt.s64.s32 	%rd22, %r13;
	mul.wide.s32 	%rd23, %r13, 4;
	add.u64 	%rd24, %rd21, %rd23;
	ld.global.f32 	%f11, [%rd24+4];
	mov.f32 	%f12, %f11;
	.loc	14	24	0
	ld.param.u64 	%rd25, [__cudaparm_kernmul_fftKxx];
	cvt.s64.s32 	%rd26, %r10;
	mul.wide.s32 	%rd27, %r10, 4;
	add.u64 	%rd28, %rd25, %rd27;
	ld.global.f32 	%f13, [%rd28+0];
	mov.f32 	%f14, %f13;
	.loc	14	25	0
	ld.param.u64 	%rd29, [__cudaparm_kernmul_fftKyy];
	cvt.s64.s32 	%rd30, %r10;
	mul.wide.s32 	%rd31, %r10, 4;
	add.u64 	%rd32, %rd29, %rd31;
	ld.global.f32 	%f15, [%rd32+0];
	mov.f32 	%f16, %f15;
	.loc	14	26	0
	ld.param.u64 	%rd33, [__cudaparm_kernmul_fftKzz];
	cvt.s64.s32 	%rd34, %r10;
	mul.wide.s32 	%rd35, %r10, 4;
	add.u64 	%rd36, %rd33, %rd35;
	ld.global.f32 	%f17, [%rd36+0];
	mov.f32 	%f18, %f17;
	.loc	14	28	0
	ld.param.u64 	%rd37, [__cudaparm_kernmul_fftKyz];
	cvt.s64.s32 	%rd38, %r10;
	mul.wide.s32 	%rd39, %r10, 4;
	add.u64 	%rd40, %rd37, %rd39;
	ld.global.f32 	%f19, [%rd40+0];
	mov.f32 	%f20, %f19;
	.loc	14	29	0
	ld.param.u64 	%rd41, [__cudaparm_kernmul_fftKxz];
	cvt.s64.s32 	%rd42, %r10;
	mul.wide.s32 	%rd43, %r10, 4;
	add.u64 	%rd44, %rd41, %rd43;
	ld.global.f32 	%f21, [%rd44+0];
	mov.f32 	%f22, %f21;
	.loc	14	30	0
	ld.param.u64 	%rd45, [__cudaparm_kernmul_fftKxy];
	cvt.s64.s32 	%rd46, %r10;
	mul.wide.s32 	%rd47, %r10, 4;
	add.u64 	%rd48, %rd45, %rd47;
	ld.global.f32 	%f23, [%rd48+0];
	mov.f32 	%f24, %f23;
	.loc	14	32	0
	mov.f32 	%f25, %f6;
	mov.f32 	%f26, %f24;
	mul.f32 	%f27, %f25, %f26;
	mov.f32 	%f28, %f2;
	mov.f32 	%f29, %f14;
	mad.f32 	%f30, %f28, %f29, %f27;
	mov.f32 	%f31, %f10;
	mov.f32 	%f32, %f22;
	mad.f32 	%f33, %f31, %f32, %f30;
	ld.param.u64 	%rd49, [__cudaparm_kernmul_fftMx];
	cvt.s64.s32 	%rd50, %r13;
	mul.wide.s32 	%rd51, %r13, 4;
	add.u64 	%rd52, %rd49, %rd51;
	st.global.f32 	[%rd52+0], %f33;
	.loc	14	33	0
	mov.f32 	%f34, %f8;
	mov.f32 	%f35, %f24;
	mul.f32 	%f36, %f34, %f35;
	mov.f32 	%f37, %f4;
	mov.f32 	%f38, %f14;
	mad.f32 	%f39, %f37, %f38, %f36;
	mov.f32 	%f40, %f12;
	mov.f32 	%f41, %f22;
	mad.f32 	%f42, %f40, %f41, %f39;
	ld.param.u64 	%rd53, [__cudaparm_kernmul_fftMx];
	cvt.s64.s32 	%rd54, %r13;
	mul.wide.s32 	%rd55, %r13, 4;
	add.u64 	%rd56, %rd53, %rd55;
	st.global.f32 	[%rd56+4], %f42;
	.loc	14	35	0
	mov.f32 	%f43, %f6;
	mov.f32 	%f44, %f16;
	mul.f32 	%f45, %f43, %f44;
	mov.f32 	%f46, %f2;
	mov.f32 	%f47, %f24;
	mad.f32 	%f48, %f46, %f47, %f45;
	mov.f32 	%f49, %f10;
	mov.f32 	%f50, %f20;
	mad.f32 	%f51, %f49, %f50, %f48;
	ld.param.u64 	%rd57, [__cudaparm_kernmul_fftMy];
	cvt.s64.s32 	%rd58, %r13;
	mul.wide.s32 	%rd59, %r13, 4;
	add.u64 	%rd60, %rd57, %rd59;
	st.global.f32 	[%rd60+0], %f51;
	.loc	14	36	0
	mov.f32 	%f52, %f8;
	mov.f32 	%f53, %f16;
	mul.f32 	%f54, %f52, %f53;
	mov.f32 	%f55, %f4;
	mov.f32 	%f56, %f24;
	mad.f32 	%f57, %f55, %f56, %f54;
	mov.f32 	%f58, %f12;
	mov.f32 	%f59, %f20;
	mad.f32 	%f60, %f58, %f59, %f57;
	ld.param.u64 	%rd61, [__cudaparm_kernmul_fftMy];
	cvt.s64.s32 	%rd62, %r13;
	mul.wide.s32 	%rd63, %r13, 4;
	add.u64 	%rd64, %rd61, %rd63;
	st.global.f32 	[%rd64+4], %f60;
	.loc	14	38	0
	mov.f32 	%f61, %f6;
	mov.f32 	%f62, %f20;
	mul.f32 	%f63, %f61, %f62;
	mov.f32 	%f64, %f2;
	mov.f32 	%f65, %f22;
	mad.f32 	%f66, %f64, %f65, %f63;
	mov.f32 	%f67, %f10;
	mov.f32 	%f68, %f18;
	mad.f32 	%f69, %f67, %f68, %f66;
	ld.param.u64 	%rd65, [__cudaparm_kernmul_fftMz];
	cvt.s64.s32 	%rd66, %r13;
	mul.wide.s32 	%rd67, %r13, 4;
	add.u64 	%rd68, %rd65, %rd67;
	st.global.f32 	[%rd68+0], %f69;
	.loc	14	39	0
	mov.f32 	%f70, %f8;
	mov.f32 	%f71, %f20;
	mul.f32 	%f72, %f70, %f71;
	mov.f32 	%f73, %f4;
	mov.f32 	%f74, %f22;
	mad.f32 	%f75, %f73, %f74, %f72;
	mov.f32 	%f76, %f12;
	mov.f32 	%f77, %f18;
	mad.f32 	%f78, %f76, %f77, %f75;
	ld.param.u64 	%rd69, [__cudaparm_kernmul_fftMz];
	cvt.s64.s32 	%rd70, %r13;
	mul.wide.s32 	%rd71, %r13, 4;
	add.u64 	%rd72, %rd69, %rd71;
	st.global.f32 	[%rd72+4], %f78;
$LDWendblock_234_3:
$L_0_1794:
$LDWendblock_234_1:
	.loc	14	41	0
	exit;
$LDWend_kernmul:
	} // kernmul

 	@@DWARF .section .debug_info, "",@progbits
	@@DWARF .byte	0xcd, 0x04, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_abbrev
	@@DWARF .4byte	0x742f0108, 0x742f706d, 0x6678706d, 0x30305f74
	@@DWARF .4byte	0x39353030, 0x305f6535, 0x30303030, 0x2d303030
	@@DWARF .4byte	0x656b5f39, 0x756d6e72, 0x70632e6c, 0x692e3370
	@@DWARF .4byte	0x6f682f00, 0x612f656d, 0x2f656e72, 0x732f6f67
	@@DWARF .4byte	0x6e2f6372, 0x6c626d69, 0x75632d65, 0x672f6562
	@@DWARF .4byte	0x702f7570, 0x6f007874, 0x636e6570, 0x2e342063
	@@DWARF .byte	0x31, 0x00, 0x04, 0x00
	@@DWARF .4byte	.debug_line
	@@DWARF .4byte	0x736e7502, 0x656e6769, 0x6e692064, 0x04070074
	@@DWARF .4byte	0x75bb0b03, 0x33746e69, 0x00b80c00, 0x0b040000
	@@DWARF .4byte	0x720078bc, 0x02000000, 0x04010023, 0x0079bc0b
	@@DWARF .4byte	0x00000072, 0x01042302, 0x7abc0b04, 0x00007200
	@@DWARF .4byte	0x08230200, 0x0b050001, 0x69750167, 0x0033746e
	@@DWARF .4byte	0x00000082, 0x0000b806, 0x8a0b0700, 0x6d696401
	@@DWARF .4byte	0x040c0033, 0x08000001, 0x78018b0b, 0x00007200
	@@DWARF .4byte	0x00230200, 0x8b0b0801, 0x72007901, 0x02000000
	@@DWARF .4byte	0x08010423, 0x7a018b0b, 0x00007200, 0x08230200
	@@DWARF .4byte	0x0b050001, 0x69640193, 0xcb00336d, 0x06000000
	@@DWARF .4byte	0x00000104, 0x746e6902, 0x06040500, 0x00000116
	@@DWARF .4byte	0x6f6c6602, 0x04007461, 0x01220904, 0x05080000
	@@DWARF .4byte	0x756f6402, 0x00656c62, 0x6c020804, 0x20676e6f
	@@DWARF .4byte	0x676e6f6c, 0x736e7520, 0x656e6769, 0x6e692064
	@@DWARF .4byte	0x08070074, 0x6e6f6c02, 0x6f6c2067, 0x6920676e
	@@DWARF .4byte	0x0500746e, 0x3e070a08, 0x61647563, 0x6e756f52
	@@DWARF .4byte	0x646f4d64, 0xd9040065, 0x0b000001, 0x75633f07
	@@DWARF .4byte	0x6f526164, 0x4e646e75, 0x65726165, 0x00007473
	@@DWARF .4byte	0x0b000000, 0x75634007, 0x6f526164, 0x5a646e75
	@@DWARF .4byte	0x006f7265, 0x00000001, 0x6341070b, 0x52616475
	@@DWARF .4byte	0x646e756f, 0x49736f50, 0x0200666e, 0x0b000000
	@@DWARF .4byte	0x75634307, 0x6f526164, 0x4d646e75, 0x6e496e69
	@@DWARF .4byte	0x00030066, 0x02000000, 0x676e6f6c, 0x746e6920
	@@DWARF .4byte	0x02080500, 0x72616863, 0x06010600, 0x000001e5
	@@DWARF .4byte	0x0001ed09, 0x03050800, 0x6c66fd0b, 0x3274616f
	@@DWARF .4byte	0x02230800, 0x0b040000, 0x220078fd, 0x02000001
	@@DWARF .4byte	0x04010023, 0x0079fd0b, 0x00000122, 0x01042302
	@@DWARF .4byte	0x730b0500, 0x6f6c6601, 0x00327461, 0x000001f9
	@@DWARF .4byte	0x0000720c, 0x023f0100, 0x050d0000, 0x14100e00
	@@DWARF .4byte	0x635f5f0d, 0x72616475, 0x32695f74, 0x5f69706f
	@@DWARF .4byte	0x02320066, 0x16090000, 0x08000001, 0x00720c05
	@@DWARF .4byte	0x6c010000, 0x0d000002, 0x220f0006, 0x06000001
	@@DWARF .4byte	0x00000122, 0x00013209, 0x02050800, 0x676e6f6c
	@@DWARF .4byte	0x736e7520, 0x656e6769, 0x6e692064, 0x08070074
	@@DWARF .byte	0x10, 0x0e, 0x09, 0x6b, 0x65, 0x72, 0x6e, 0x6d
	@@DWARF .byte	0x75, 0x6c, 0x00, 0x01, 0x01, 0x00
	@@DWARF .quad	$LDWbegin_kernmul
	@@DWARF .quad	$LDWend_kernmul
	@@DWARF .byte	0x11, 0x0e, 0x09, 0x66, 0x66, 0x74, 0x4d, 0x78
	@@DWARF .byte	0x00, 0x2b, 0x01, 0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_kernmul_fftMx
	@@DWARF .4byte	0x090e1107, 0x4d746666, 0x012b0079, 0x03090000
	@@DWARF .quad	__cudaparm_kernmul_fftMy
	@@DWARF .4byte	0x090e1107, 0x4d746666, 0x012b007a, 0x03090000
	@@DWARF .quad	__cudaparm_kernmul_fftMz
	@@DWARF .4byte	0x090e1107, 0x4b746666, 0x2b007878, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul_fftKxx
	@@DWARF .4byte	0x090e1107, 0x4b746666, 0x2b007979, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul_fftKyy
	@@DWARF .4byte	0x090e1107, 0x4b746666, 0x2b007a7a, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul_fftKzz
	@@DWARF .4byte	0x090e1107, 0x4b746666, 0x2b007a79, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul_fftKyz
	@@DWARF .4byte	0x090e1107, 0x4b746666, 0x2b007a78, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul_fftKxz
	@@DWARF .4byte	0x090e1107, 0x4b746666, 0x2b007978, 0x09000001
	@@DWARF .byte	0x03
	@@DWARF .quad	__cudaparm_kernmul_fftKxy
	@@DWARF .byte	0x07, 0x11, 0x0e, 0x09, 0x4e, 0x00, 0x16, 0x01
	@@DWARF .byte	0x00, 0x00, 0x09, 0x03
	@@DWARF .quad	__cudaparm_kernmul_N
	@@DWARF .byte	0x07, 0x12
	@@DWARF .quad	$LDWbeginblock_234_1
	@@DWARF .quad	$LDWendblock_234_1
	@@DWARF .4byte	0x690b0e13, 0x00011600, 0xb0900600, 0x02abc8e2
	@@DWARF .4byte	0x0c0e1302, 0x01160065, 0x90060000, 0xabc8e2b3
	@@DWARF .byte	0x02, 0x02, 0x12
	@@DWARF .quad	$LDWbeginblock_234_3
	@@DWARF .quad	$LDWendblock_234_3
	@@DWARF .4byte	0x720f0e13, 0x00784d65, 0x00000122, 0xccb29005
	@@DWARF .4byte	0x13020195, 0x6d69100e, 0x2200784d, 0x05000001
	@@DWARF .4byte	0x95ccb490, 0x0e130201, 0x4d657212, 0x01220079
	@@DWARF .4byte	0x90050000, 0x0195ccb6, 0x130e1302, 0x794d6d69
	@@DWARF .4byte	0x00012200, 0xb8900500, 0x020195cc, 0x72150e13
	@@DWARF .4byte	0x007a4d65, 0x00000122, 0xe2b09006, 0x0202ab98
	@@DWARF .4byte	0x69160e13, 0x007a4d6d, 0x00000122, 0xe2b29006
	@@DWARF .4byte	0x0202ab98, 0x4b180e13, 0x22007878, 0x06000001
	@@DWARF .4byte	0x98e2b490, 0x130202ab, 0x794b190e, 0x01220079
	@@DWARF .4byte	0x90060000, 0xab98e2b6, 0x0e130202, 0x7a7a4b1a
	@@DWARF .4byte	0x00012200, 0xb8900600, 0x02ab98e2, 0x1c0e1302
	@@DWARF .4byte	0x007a794b, 0x00000122, 0xe4b09006, 0x0202ab98
	@@DWARF .4byte	0x4b1d0e13, 0x22007a78, 0x06000001, 0x98e4b290
	@@DWARF .4byte	0x130202ab, 0x784b1e0e, 0x01220079, 0x90060000
	@@DWARF .byte	0xb4, 0xe4, 0x98, 0xab, 0x02, 0x02, 0x00, 0x00
	@@DWARF .byte	0x00, 0x00, 0x00

 	@@DWARF .section .debug_pubnames, "",@progbits
	@@DWARF .byte	0x1a, 0x00, 0x00, 0x00, 0x02, 0x00
	@@DWARF .4byte	.debug_info
	@@DWARF .4byte	0x000004d1, 0x00000292, 0x6e72656b, 0x006c756d
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
