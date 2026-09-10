//go:build darwin && arm64 && !cgo
// +build darwin,arm64,!cgo

package cufft

import "github.com/mumax/3/cuda/cu"

const metalCgoRequired = "metal cufft: Apple Silicon FFT requires CGO_ENABLED=1"

type Handle uintptr

func Plan1d(nx int, typ Type, batch int) Handle {
	panic(metalCgoRequired)
}

func Plan2d(nx, ny int, typ Type) Handle {
	panic(metalCgoRequired)
}

func Plan3d(nx, ny, nz int, typ Type) Handle {
	panic(metalCgoRequired)
}

func PlanMany(n []int, inembed []int, istride int, oembed []int, ostride int, typ Type, batch int) Handle {
	panic(metalCgoRequired)
}

func (plan Handle) ExecC2C(idata, odata cu.DevicePtr, direction int) {
	panic(metalCgoRequired)
}

func (plan Handle) ExecR2C(idata, odata cu.DevicePtr) {
	panic(metalCgoRequired)
}

func (plan Handle) ExecC2R(idata, odata cu.DevicePtr) {
	panic(metalCgoRequired)
}

func (plan Handle) ExecZ2Z(idata, odata cu.DevicePtr, direction int) {
	panic(metalCgoRequired)
}

func (plan Handle) ExecD2Z(idata, odata cu.DevicePtr) {
	panic(metalCgoRequired)
}

func (plan Handle) ExecZ2D(idata, odata cu.DevicePtr) {
	panic(metalCgoRequired)
}

func (plan *Handle) Destroy() {
	panic(metalCgoRequired)
}

func (plan Handle) SetStream(stream cu.Stream) {
	panic(metalCgoRequired)
}
