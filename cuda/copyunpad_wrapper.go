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

// CUDA handle for copyunpad kernel
var copyunpad_code cu.Function

// Stores the arguments for copyunpad kernel invocation
type copyunpad_args_t struct {
	arg_dst unsafe.Pointer
	arg_Dx  int
	arg_Dy  int
	arg_Dz  int
	arg_src unsafe.Pointer
	arg_Sx  int
	arg_Sy  int
	arg_Sz  int
	argptr  [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for copyunpad kernel invocation
var copyunpad_args copyunpad_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	copyunpad_args.argptr[0] = unsafe.Pointer(&copyunpad_args.arg_dst)
	copyunpad_args.argptr[1] = unsafe.Pointer(&copyunpad_args.arg_Dx)
	copyunpad_args.argptr[2] = unsafe.Pointer(&copyunpad_args.arg_Dy)
	copyunpad_args.argptr[3] = unsafe.Pointer(&copyunpad_args.arg_Dz)
	copyunpad_args.argptr[4] = unsafe.Pointer(&copyunpad_args.arg_src)
	copyunpad_args.argptr[5] = unsafe.Pointer(&copyunpad_args.arg_Sx)
	copyunpad_args.argptr[6] = unsafe.Pointer(&copyunpad_args.arg_Sy)
	copyunpad_args.argptr[7] = unsafe.Pointer(&copyunpad_args.arg_Sz)
}

// Wrapper for copyunpad CUDA kernel, asynchronous.
func k_copyunpad_async(dst unsafe.Pointer, Dx int, Dy int, Dz int, src unsafe.Pointer, Sx int, Sy int, Sz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("copyunpad")
	}

	copyunpad_args.Lock()
	defer copyunpad_args.Unlock()

	if copyunpad_code == 0 {
		copyunpad_code = fatbinLoad(copyunpad_map, "copyunpad")
	}

	copyunpad_args.arg_dst = dst
	copyunpad_args.arg_Dx = Dx
	copyunpad_args.arg_Dy = Dy
	copyunpad_args.arg_Dz = Dz
	copyunpad_args.arg_src = src
	copyunpad_args.arg_Sx = Sx
	copyunpad_args.arg_Sy = Sy
	copyunpad_args.arg_Sz = Sz

	args := copyunpad_args.argptr[:]
	cu.LaunchKernel(copyunpad_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("copyunpad")
	}
}

// maps compute capability on PTX code for copyunpad kernel.
var copyunpad_map = map[int]string{0: "",
	50: copyunpad_ptx_50,
	52: copyunpad_ptx_52,
	53: copyunpad_ptx_53,
	60: copyunpad_ptx_60,
	61: copyunpad_ptx_61,
	62: copyunpad_ptx_62,
	70: copyunpad_ptx_70,
	72: copyunpad_ptx_72,
	75: copyunpad_ptx_75,
	80: copyunpad_ptx_80,
	86: copyunpad_ptx_86,
	87: copyunpad_ptx_87,
	89: copyunpad_ptx_89,
	90: copyunpad_ptx_90}

// copyunpad PTX code for various compute capabilities.
const (
	copyunpad_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_86 = `
.version 8.5
.target sm_86
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_87 = `
.version 8.5
.target sm_87
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_89 = `
.version 8.5
.target sm_89
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	copyunpad_ptx_90 = `
.version 8.5
.target sm_90
.address_size 64

	// .globl	copyunpad

.visible .entry copyunpad(
	.param .u64 copyunpad_param_0,
	.param .u32 copyunpad_param_1,
	.param .u32 copyunpad_param_2,
	.param .u32 copyunpad_param_3,
	.param .u64 copyunpad_param_4,
	.param .u32 copyunpad_param_5,
	.param .u32 copyunpad_param_6,
	.param .u32 copyunpad_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [copyunpad_param_0];
	ld.param.u32 	%r4, [copyunpad_param_1];
	ld.param.u32 	%r5, [copyunpad_param_2];
	ld.param.u32 	%r8, [copyunpad_param_3];
	ld.param.u64 	%rd2, [copyunpad_param_4];
	ld.param.u32 	%r6, [copyunpad_param_5];
	ld.param.u32 	%r7, [copyunpad_param_6];
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %ntid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r8;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	mul.wide.s32 	%rd4, %r19, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r21, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
)
