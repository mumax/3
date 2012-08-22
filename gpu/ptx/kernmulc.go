package ptx

const KERNMULC = `
	.version 1.4
	.target sm_10, map_f64_to_f32
	// compiled with /usr/local/cuda/open64/lib//be
	// nvopencc 4.1 built on 2012-04-05

	//-----------------------------------------------------------
	// Compiling /tmp/tmpxft_00005437_00000000-9_kernmulc.cpp3.i (/tmp/ccBI#.Kg3UJr)
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
	.file	2	"/tmp/tmpxft_00005437_00000000-8_kernmulc.cudafe2.gpu"
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
	.file	14	"kernmulc.cu"
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


	.entry kernmulC (
		.param .u64 __cudaparm_kernmulC_Mx,
		.param .u64 __cudaparm_kernmulC_My,
		.param .u64 __cudaparm_kernmulC_Mz,
		.param .u64 __cudaparm_kernmulC_Kxx,
		.param .u64 __cudaparm_kernmulC_Kyy,
		.param .u64 __cudaparm_kernmulC_Kzz,
		.param .u64 __cudaparm_kernmulC_Kyz,
		.param .u64 __cudaparm_kernmulC_Kxz,
		.param .u64 __cudaparm_kernmulC_Kxy,
		.param .u64 __cudaparm_kernmulC_Kzy,
		.param .u64 __cudaparm_kernmulC_Kzx,
		.param .u64 __cudaparm_kernmulC_Kyx,
		.param .s32 __cudaparm_kernmulC_N)
	{
	.reg .u16 %rh<4>;
	.reg .u32 %r<11>;
	.reg .u64 %rd<28>;
	.reg .f32 %f<83>;
	.reg .pred %p<3>;
	.loc	14	14	0
$LDWbegin_kernmulC:
	mov.u16 	%rh1, %nctaid.x;
	mov.u16 	%rh2, %ctaid.y;
	mul.wide.u16 	%r1, %rh1, %rh2;
	cvt.u32.u16 	%r2, %ctaid.x;
	add.u32 	%r3, %r2, %r1;
	cvt.u32.u16 	%r4, %ntid.x;
	mul.lo.u32 	%r5, %r4, %r3;
	cvt.u32.u16 	%r6, %tid.x;
	add.u32 	%r7, %r6, %r5;
	ld.param.s32 	%r8, [__cudaparm_kernmulC_N];
	setp.le.s32 	%p1, %r8, %r7;
	@%p1 bra 	$Lt_0_1026;
	.loc	14	20	0
	mul.lo.s32 	%r9, %r7, 2;
	cvt.s64.s32 	%rd1, %r9;
	mul.wide.s32 	%rd2, %r9, 4;
	ld.param.u64 	%rd3, [__cudaparm_kernmulC_Mx];
	add.u64 	%rd4, %rd3, %rd2;
	ld.global.f32 	%f1, [%rd4+0];
	.loc	14	21	0
	ld.global.f32 	%f2, [%rd4+4];
	.loc	14	23	0
	ld.param.u64 	%rd5, [__cudaparm_kernmulC_My];
	add.u64 	%rd6, %rd5, %rd2;
	ld.global.f32 	%f3, [%rd6+0];
	.loc	14	24	0
	ld.global.f32 	%f4, [%rd6+4];
	.loc	14	26	0
	ld.param.u64 	%rd7, [__cudaparm_kernmulC_Mz];
	add.u64 	%rd8, %rd7, %rd2;
	ld.global.f32 	%f5, [%rd8+0];
	.loc	14	27	0
	ld.global.f32 	%f6, [%rd8+4];
	.loc	14	29	0
	ld.param.u64 	%rd9, [__cudaparm_kernmulC_Kxx];
	add.u64 	%rd10, %rd9, %rd2;
	ld.global.f32 	%f7, [%rd10+0];
	.loc	14	30	0
	ld.param.u64 	%rd11, [__cudaparm_kernmulC_Kyy];
	add.u64 	%rd12, %rd11, %rd2;
	ld.global.f32 	%f8, [%rd12+0];
	.loc	14	31	0
	ld.param.u64 	%rd13, [__cudaparm_kernmulC_Kzz];
	add.u64 	%rd14, %rd13, %rd2;
	ld.global.f32 	%f9, [%rd14+0];
	.loc	14	32	0
	ld.global.f32 	%f10, [%rd10+4];
	.loc	14	33	0
	ld.global.f32 	%f11, [%rd12+4];
	.loc	14	34	0
	ld.global.f32 	%f12, [%rd14+4];
	.loc	14	36	0
	ld.param.u64 	%rd15, [__cudaparm_kernmulC_Kyz];
	add.u64 	%rd16, %rd15, %rd2;
	ld.global.f32 	%f13, [%rd16+0];
	.loc	14	37	0
	ld.param.u64 	%rd17, [__cudaparm_kernmulC_Kxz];
	add.u64 	%rd18, %rd17, %rd2;
	ld.global.f32 	%f14, [%rd18+0];
	.loc	14	38	0
	ld.param.u64 	%rd19, [__cudaparm_kernmulC_Kxy];
	add.u64 	%rd20, %rd19, %rd2;
	ld.global.f32 	%f15, [%rd20+0];
	.loc	14	39	0
	ld.global.f32 	%f16, [%rd16+4];
	.loc	14	40	0
	ld.global.f32 	%f17, [%rd18+4];
	.loc	14	41	0
	ld.global.f32 	%f18, [%rd20+4];
	.loc	14	43	0
	ld.param.u64 	%rd21, [__cudaparm_kernmulC_Kzy];
	add.u64 	%rd22, %rd21, %rd2;
	ld.global.f32 	%f19, [%rd22+0];
	.loc	14	44	0
	ld.param.u64 	%rd23, [__cudaparm_kernmulC_Kzx];
	add.u64 	%rd24, %rd23, %rd2;
	ld.global.f32 	%f20, [%rd24+0];
	.loc	14	45	0
	ld.param.u64 	%rd25, [__cudaparm_kernmulC_Kyx];
	add.u64 	%rd26, %rd25, %rd2;
	ld.global.f32 	%f21, [%rd26+0];
	.loc	14	46	0
	ld.global.f32 	%f22, [%rd22+4];
	.loc	14	47	0
	ld.global.f32 	%f23, [%rd24+4];
	.loc	14	48	0
	ld.global.f32 	%f24, [%rd26+4];
	.loc	14	50	0
	mul.f32 	%f25, %f2, %f10;
	mul.f32 	%f26, %f1, %f7;
	sub.f32 	%f27, %f26, %f25;
	mul.f32 	%f28, %f4, %f18;
	mul.f32 	%f29, %f3, %f15;
	sub.f32 	%f30, %f29, %f28;
	add.f32 	%f31, %f27, %f30;
	mul.f32 	%f32, %f6, %f17;
	mul.f32 	%f33, %f5, %f14;
	sub.f32 	%f34, %f33, %f32;
	add.f32 	%f35, %f31, %f34;
	st.global.f32 	[%rd4+0], %f35;
	.loc	14	51	0
	mul.f32 	%f36, %f2, %f7;
	mad.f32 	%f37, %f1, %f10, %f36;
	mul.f32 	%f38, %f4, %f15;
	mad.f32 	%f39, %f3, %f18, %f38;
	add.f32 	%f40, %f37, %f39;
	mul.f32 	%f41, %f6, %f14;
	mad.f32 	%f42, %f5, %f17, %f41;
	add.f32 	%f43, %f40, %f42;
	st.global.f32 	[%rd4+4], %f43;
	.loc	14	53	0
	mul.f32 	%f44, %f2, %f24;
	mul.f32 	%f45, %f1, %f21;
	sub.f32 	%f46, %f45, %f44;
	mul.f32 	%f47, %f4, %f11;
	mul.f32 	%f48, %f3, %f8;
	sub.f32 	%f49, %f48, %f47;
	add.f32 	%f50, %f46, %f49;
	mul.f32 	%f51, %f6, %f16;
	mul.f32 	%f52, %f5, %f13;
	sub.f32 	%f53, %f52, %f51;
	add.f32 	%f54, %f50, %f53;
	st.global.f32 	[%rd6+0], %f54;
	.loc	14	54	0
	mul.f32 	%f55, %f2, %f21;
	mad.f32 	%f56, %f1, %f24, %f55;
	mul.f32 	%f57, %f4, %f8;
	mad.f32 	%f58, %f3, %f11, %f57;
	add.f32 	%f59, %f56, %f58;
	mul.f32 	%f60, %f6, %f13;
	mad.f32 	%f61, %f5, %f16, %f60;
	add.f32 	%f62, %f59, %f61;
	st.global.f32 	[%rd6+4], %f62;
	.loc	14	56	0
	mul.f32 	%f63, %f2, %f23;
	mul.f32 	%f64, %f1, %f20;
	sub.f32 	%f65, %f64, %f63;
	mul.f32 	%f66, %f4, %f22;
	mul.f32 	%f67, %f3, %f19;
	sub.f32 	%f68, %f67, %f66;
	add.f32 	%f69, %f65, %f68;
	mul.f32 	%f70, %f6, %f12;
	mul.f32 	%f71, %f5, %f9;
	sub.f32 	%f72, %f71, %f70;
	add.f32 	%f73, %f69, %f72;
	st.global.f32 	[%rd8+0], %f73;
	.loc	14	57	0
	mul.f32 	%f74, %f2, %f20;
	mad.f32 	%f75, %f1, %f23, %f74;
	mul.f32 	%f76, %f4, %f19;
	mad.f32 	%f77, %f3, %f22, %f76;
	add.f32 	%f78, %f75, %f77;
	mul.f32 	%f79, %f6, %f9;
	mad.f32 	%f80, %f5, %f12, %f79;
	add.f32 	%f81, %f78, %f80;
	st.global.f32 	[%rd8+4], %f81;
$Lt_0_1026:
	.loc	14	59	0
	exit;
$LDWend_kernmulC:
	} // kernmulC

`
