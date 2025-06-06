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

// CUDA handle for regionadds kernel
var regionadds_code cu.Function

// Stores the arguments for regionadds kernel invocation
type regionadds_args_t struct {
	arg_dst     unsafe.Pointer
	arg_LUT     unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_N       int
	argptr      [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for regionadds kernel invocation
var regionadds_args regionadds_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	regionadds_args.argptr[0] = unsafe.Pointer(&regionadds_args.arg_dst)
	regionadds_args.argptr[1] = unsafe.Pointer(&regionadds_args.arg_LUT)
	regionadds_args.argptr[2] = unsafe.Pointer(&regionadds_args.arg_regions)
	regionadds_args.argptr[3] = unsafe.Pointer(&regionadds_args.arg_N)
}

// Wrapper for regionadds CUDA kernel, asynchronous.
func k_regionadds_async(dst unsafe.Pointer, LUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("regionadds")
	}

	regionadds_args.Lock()
	defer regionadds_args.Unlock()

	if regionadds_code == 0 {
		regionadds_code = fatbinLoad(regionadds_map, "regionadds")
	}

	regionadds_args.arg_dst = dst
	regionadds_args.arg_LUT = LUT
	regionadds_args.arg_regions = regions
	regionadds_args.arg_N = N

	args := regionadds_args.argptr[:]
	cu.LaunchKernel(regionadds_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("regionadds")
	}
}

// maps compute capability on PTX code for regionadds kernel.
var regionadds_map = map[int]string{0: "",
	50: regionadds_ptx_50,
	52: regionadds_ptx_52,
	53: regionadds_ptx_53,
	60: regionadds_ptx_60,
	61: regionadds_ptx_61,
	62: regionadds_ptx_62,
	70: regionadds_ptx_70,
	72: regionadds_ptx_72,
	75: regionadds_ptx_75,
	80: regionadds_ptx_80,
	86: regionadds_ptx_86,
	87: regionadds_ptx_87,
	89: regionadds_ptx_89,
	90: regionadds_ptx_90}

// regionadds PTX code for various compute capabilities.
const (
	regionadds_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_86 = `
.version 8.5
.target sm_86
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_87 = `
.version 8.5
.target sm_87
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_89 = `
.version 8.5
.target sm_89
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	regionadds_ptx_90 = `
.version 8.5
.target sm_90
.address_size 64

	// .globl	regionadds

.visible .entry regionadds(
	.param .u64 regionadds_param_0,
	.param .u64 regionadds_param_1,
	.param .u64 regionadds_param_2,
	.param .u32 regionadds_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [regionadds_param_0];
	ld.param.u64 	%rd2, [regionadds_param_1];
	ld.param.u64 	%rd3, [regionadds_param_2];
	ld.param.u32 	%r2, [regionadds_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	cvta.to.global.u64 	%rd7, %rd2;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd8, %r10, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
)
