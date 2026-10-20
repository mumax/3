//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package cufft

import (
	"fmt"

	"github.com/mumax/3/cuda/cu"
	metalfft "github.com/mumax/3/cuda/metal/fft"
)

// Handle is an opaque retained MPSGraph plan.
type Handle uintptr

func Plan1d(nx int, typ Type, batch int) Handle {
	return newMetalPlan([]int{nx}, typ, batch)
}

func Plan2d(nx, ny int, typ Type) Handle {
	return newMetalPlan([]int{nx, ny}, typ, 1)
}

func Plan3d(nx, ny, nz int, typ Type) Handle {
	return newMetalPlan([]int{nx, ny, nz}, typ, 1)
}

// PlanMany supports the contiguous subset used by mumax3. Non-unit strides or
// explicit embedding would require a gather/scatter graph and are rejected
// rather than silently producing a differently laid-out transform.
func PlanMany(n []int, inembed []int, istride int, oembed []int, ostride int, typ Type, batch int) Handle {
	if len(n) == 0 {
		panic("metal cufft: PlanMany dimensions are empty")
	}
	if istride != 1 || ostride != 1 {
		panic(fmt.Sprintf("metal cufft: PlanMany requires unit strides, got input=%d output=%d", istride, ostride))
	}
	if !sameOrNil(inembed, n) || !sameOrNil(oembed, n) {
		panic("metal cufft: PlanMany explicit embedding is not supported")
	}
	return newMetalPlan(n, typ, batch)
}

func newMetalPlan(dimensions []int, typ Type, batch int) Handle {
	transform := metalfft.Transform(typ)
	switch typ {
	case R2C, C2R, C2C:
	default:
		panic(fmt.Sprintf("metal cufft: %s is unsupported; mumax3 uses single-precision R2C/C2R", typ))
	}
	layout, err := metalfft.NewLayout(dimensions, batch)
	if err != nil {
		panic(err)
	}
	handle, err := metalfft.CreatePlan(layout, transform)
	if err != nil {
		panic(err)
	}
	return Handle(handle)
}

func (plan Handle) ExecC2C(idata, odata cu.DevicePtr, direction int) {
	mustMetalFFT(plan, idata, odata, direction)
}

func (plan Handle) ExecR2C(idata, odata cu.DevicePtr) {
	mustMetalFFT(plan, idata, odata, FORWARD)
}

func (plan Handle) ExecC2R(idata, odata cu.DevicePtr) {
	mustMetalFFT(plan, idata, odata, INVERSE)
}

func (plan Handle) ExecZ2Z(idata, odata cu.DevicePtr, direction int) {
	panic("metal cufft: double-precision Z2Z is unsupported on Apple GPUs")
}

func (plan Handle) ExecD2Z(idata, odata cu.DevicePtr) {
	panic("metal cufft: double-precision D2Z is unsupported on Apple GPUs")
}

func (plan Handle) ExecZ2D(idata, odata cu.DevicePtr) {
	panic("metal cufft: double-precision Z2D is unsupported on Apple GPUs")
}

func mustMetalFFT(plan Handle, idata, odata cu.DevicePtr, direction int) {
	if err := metalfft.Execute(uintptr(plan), uintptr(idata), uintptr(odata), direction); err != nil {
		panic(err)
	}
}

func (plan *Handle) Destroy() {
	if plan == nil || *plan == 0 {
		return
	}
	if err := metalfft.DestroyPlan(uintptr(*plan)); err != nil {
		panic(err)
	}
	*plan = 0
}

// The Metal runtime deliberately exposes one serial command queue. SetStream
// is retained for API compatibility; all plans append to that same queue.
func (plan Handle) SetStream(stream cu.Stream) {
	if plan == 0 {
		panic("metal cufft: SetStream called on a destroyed plan")
	}
	_ = stream
}

func sameOrNil(got, want []int) bool {
	if got == nil {
		return true
	}
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
