package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for crop kernel
var crop_code cu.Function

// Stores the arguments for crop kernel invocation
type crop_args_t struct {
	arg_dst  unsafe.Pointer
	arg_Dx   int
	arg_Dy   int
	arg_Dz   int
	arg_src  unsafe.Pointer
	arg_Sx   int
	arg_Sy   int
	arg_Sz   int
	arg_Offx int
	arg_Offy int
	arg_Offz int
	argptr   [11]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for crop kernel invocation
var crop_args crop_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	crop_args.argptr[0] = unsafe.Pointer(&crop_args.arg_dst)
	crop_args.argptr[1] = unsafe.Pointer(&crop_args.arg_Dx)
	crop_args.argptr[2] = unsafe.Pointer(&crop_args.arg_Dy)
	crop_args.argptr[3] = unsafe.Pointer(&crop_args.arg_Dz)
	crop_args.argptr[4] = unsafe.Pointer(&crop_args.arg_src)
	crop_args.argptr[5] = unsafe.Pointer(&crop_args.arg_Sx)
	crop_args.argptr[6] = unsafe.Pointer(&crop_args.arg_Sy)
	crop_args.argptr[7] = unsafe.Pointer(&crop_args.arg_Sz)
	crop_args.argptr[8] = unsafe.Pointer(&crop_args.arg_Offx)
	crop_args.argptr[9] = unsafe.Pointer(&crop_args.arg_Offy)
	crop_args.argptr[10] = unsafe.Pointer(&crop_args.arg_Offz)
}

// Wrapper for crop CUDA kernel, asynchronous.
func k_crop_async(dst unsafe.Pointer, Dx int, Dy int, Dz int, src unsafe.Pointer, Sx int, Sy int, Sz int, Offx int, Offy int, Offz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("crop")
	}

	crop_args.Lock()
	defer crop_args.Unlock()

	if crop_code == 0 {
		crop_code = fatbinLoad(crop_map, "crop")
	}

	crop_args.arg_dst = dst
	crop_args.arg_Dx = Dx
	crop_args.arg_Dy = Dy
	crop_args.arg_Dz = Dz
	crop_args.arg_src = src
	crop_args.arg_Sx = Sx
	crop_args.arg_Sy = Sy
	crop_args.arg_Sz = Sz
	crop_args.arg_Offx = Offx
	crop_args.arg_Offy = Offy
	crop_args.arg_Offz = Offz

	args := crop_args.argptr[:]
	cu.LaunchKernel(crop_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("crop")
	}
}

// maps compute capability on PTX code for crop kernel.
var crop_map = map[int]string{0: "",
	50: crop_ptx_50,
	52: crop_ptx_52,
	53: crop_ptx_53,
	60: crop_ptx_60,
	61: crop_ptx_61,
	62: crop_ptx_62,
	70: crop_ptx_70,
	72: crop_ptx_72,
	75: crop_ptx_75,
	80: crop_ptx_80,
	86: crop_ptx_86,
	87: crop_ptx_87,
	89: crop_ptx_89,
	90: crop_ptx_90}

// crop PTX code for various compute capabilities.
const (
	crop_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_86 = `
.version 8.5
.target sm_86
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_87 = `
.version 8.5
.target sm_87
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_89 = `
.version 8.5
.target sm_89
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	crop_ptx_90 = `
.version 8.5
.target sm_90
.address_size 64

	// .globl	crop

.visible .entry crop(
	.param .u64 crop_param_0,
	.param .u32 crop_param_1,
	.param .u32 crop_param_2,
	.param .u32 crop_param_3,
	.param .u64 crop_param_4,
	.param .u32 crop_param_5,
	.param .u32 crop_param_6,
	.param .u32 crop_param_7,
	.param .u32 crop_param_8,
	.param .u32 crop_param_9,
	.param .u32 crop_param_10
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [crop_param_0];
	ld.param.u32 	%r4, [crop_param_1];
	ld.param.u32 	%r5, [crop_param_2];
	ld.param.u32 	%r11, [crop_param_3];
	ld.param.u64 	%rd2, [crop_param_4];
	ld.param.u32 	%r6, [crop_param_5];
	ld.param.u32 	%r7, [crop_param_6];
	ld.param.u32 	%r8, [crop_param_8];
	ld.param.u32 	%r9, [crop_param_9];
	ld.param.u32 	%r10, [crop_param_10];
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %ntid.x;
	mov.u32 	%r14, %tid.x;
	mad.lo.s32 	%r1, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.y;
	mov.u32 	%r16, %ctaid.y;
	mov.u32 	%r17, %tid.y;
	mad.lo.s32 	%r2, %r16, %r15, %r17;
	mov.u32 	%r18, %ntid.z;
	mov.u32 	%r19, %ctaid.z;
	mov.u32 	%r20, %tid.z;
	mad.lo.s32 	%r3, %r19, %r18, %r20;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r11;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	add.s32 	%r21, %r3, %r10;
	add.s32 	%r22, %r2, %r9;
	mad.lo.s32 	%r23, %r21, %r7, %r22;
	add.s32 	%r24, %r1, %r8;
	mad.lo.s32 	%r25, %r23, %r6, %r24;
	mul.wide.s32 	%rd4, %r25, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r27, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
)
