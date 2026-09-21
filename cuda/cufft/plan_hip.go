//go:build hip

// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package cufft

//#include <hipfft/hipfft.h>
import "C"

import (
	"unsafe"

	"github.com/mumax/3/cuda/cu"
)

// FFT plan handle, reference type to a plan
type Handle uintptr

// 1D FFT plan
func Plan1d(nx int, typ Type, batch int) Handle {
	var handle C.hipfftHandle
	err := Result(C.hipfftPlan1d(
		&handle,
		C.int(nx),
		C.hipfftType(typ),
		C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(uintptr(unsafe.Pointer(handle)))
}

// 2D FFT plan
func Plan2d(nx, ny int, typ Type) Handle {
	var handle C.hipfftHandle
	err := Result(C.hipfftPlan2d(
		&handle,
		C.int(nx),
		C.int(ny),
		C.hipfftType(typ)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(uintptr(unsafe.Pointer(handle)))
}

// 3D FFT plan
func Plan3d(nx, ny, nz int, typ Type) Handle {
	var handle C.hipfftHandle
	err := Result(C.hipfftPlan3d(
		&handle,
		C.int(nx),
		C.int(ny),
		C.int(nz),
		C.hipfftType(typ)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(uintptr(unsafe.Pointer(handle)))
}

// 1D,2D or 3D FFT plan
func PlanMany(n []int, inembed []int, istride int, oembed []int, ostride int, typ Type, batch int) Handle {
	var handle C.hipfftHandle

	NULL := (*C.int)(unsafe.Pointer(uintptr(0)))

	inembedptr := NULL
	idist := 0
	if inembed != nil {
		inembedptr = (*C.int)(unsafe.Pointer(&inembed[0]))
		idist = inembed[0]
	}

	oembedptr := NULL
	odist := 0
	if oembed != nil {
		oembedptr = (*C.int)(unsafe.Pointer(&oembed[0]))
		odist = oembed[0]
	}

	err := Result(C.hipfftPlanMany(
		&handle,
		C.int(len(n)),                   // rank
		(*C.int)(unsafe.Pointer(&n[0])), // n
		inembedptr,
		C.int(istride),
		C.int(idist),
		oembedptr,
		C.int(ostride),
		C.int(odist),
		C.hipfftType(typ),
		C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(uintptr(unsafe.Pointer(handle)))
}

func (plan Handle) handle() C.hipfftHandle {
	return C.hipfftHandle(unsafe.Pointer(uintptr(plan)))
}

// Execute Complex-to-Complex plan
func (plan Handle) ExecC2C(idata, odata cu.DevicePtr, direction int) {
	err := Result(C.hipfftExecC2C(
		plan.handle(),
		(*C.hipfftComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.hipfftComplex)(unsafe.Pointer(uintptr(odata))),
		C.int(direction)))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Real-to-Complex plan
func (plan Handle) ExecR2C(idata, odata cu.DevicePtr) {
	err := Result(C.hipfftExecR2C(
		plan.handle(),
		(*C.hipfftReal)(unsafe.Pointer(uintptr(idata))),
		(*C.hipfftComplex)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Complex-to-Real plan
func (plan Handle) ExecC2R(idata, odata cu.DevicePtr) {
	err := Result(C.hipfftExecC2R(
		plan.handle(),
		(*C.hipfftComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.hipfftReal)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Double Complex-to-Complex plan
func (plan Handle) ExecZ2Z(idata, odata cu.DevicePtr, direction int) {
	err := Result(C.hipfftExecZ2Z(
		plan.handle(),
		(*C.hipfftDoubleComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.hipfftDoubleComplex)(unsafe.Pointer(uintptr(odata))),
		C.int(direction)))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Double Real-to-Complex plan
func (plan Handle) ExecD2Z(idata, odata cu.DevicePtr) {
	err := Result(C.hipfftExecD2Z(
		plan.handle(),
		(*C.hipfftDoubleReal)(unsafe.Pointer(uintptr(idata))),
		(*C.hipfftDoubleComplex)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Double Complex-to-Real plan
func (plan Handle) ExecZ2D(idata, odata cu.DevicePtr) {
	err := Result(C.hipfftExecZ2D(
		plan.handle(),
		(*C.hipfftDoubleComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.hipfftDoubleReal)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys the plan.
func (plan *Handle) Destroy() {
	err := Result(C.hipfftDestroy(plan.handle()))
	*plan = 0 // make sure plan is not used anymore
	if err != SUCCESS {
		panic(err)
	}
}

// Sets the stream for this plan
func (plan Handle) SetStream(stream cu.Stream) {
	err := Result(C.hipfftSetStream(
		plan.handle(),
		C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}
