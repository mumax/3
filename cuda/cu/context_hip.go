//go:build hip

package cu

// This file implements HIP driver context management

//#include <hip/hip_runtime.h>
import "C"
import "unsafe"

// HIP context.
type Context uintptr

// Create a HIP context.
func CtxCreate(flags uint, dev Device) Context {
	var ctx C.hipCtx_t
	err := Result(C.hipCtxCreate(&ctx, C.uint(flags), C.hipDevice_t(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return Context(uintptr(unsafe.Pointer(ctx)))
}

// Destroys the HIP context specified by ctx. If the context usage count is not equal to 1, or the context is current to any CPU thread other than the current one, this function fails. Floating contexts may be destroyed by this function.
func CtxDestroy(ctx *Context) {
	err := Result(C.hipCtxDestroy(C.hipCtx_t(unsafe.Pointer(uintptr(*ctx)))))
	*ctx = 0
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys the HIP context.
func (ctx *Context) Destroy() {
	CtxDestroy(ctx)
}

// Returns the API version to create the context.
func CtxGetApiVersion(ctx Context) (version int) {
	var cversion C.uint
	err := Result(C.hipCtxGetApiVersion(C.hipCtx_t(unsafe.Pointer(uintptr(ctx))), &cversion))
	if err != SUCCESS {
		panic(err)
	}
	version = int(cversion)
	return
}

// Returns the API version to create the context.
func (ctx Context) ApiVersion() (version int) {
	return CtxGetApiVersion(ctx)
}

// Gets the current active context.
func CtxGetCurrent() Context {
	var ctx C.hipCtx_t
	err := Result(C.hipCtxGetCurrent(&ctx))
	if err != SUCCESS {
		panic(err)
	}
	return Context(uintptr(unsafe.Pointer(ctx)))
}

// Returns the ordinal of the current context's device.
func CtxGetDevice() Device {
	var dev C.hipDevice_t
	err := Result(C.hipCtxGetDevice(&dev))
	if err != SUCCESS {
		panic(err)
	}
	return Device(dev)
}

// Sets the current active context.
func CtxSetCurrent(ctx Context) {
	err := Result(C.hipCtxSetCurrent(C.hipCtx_t(unsafe.Pointer(uintptr(ctx)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Sets the current active context.
func (ctx Context) SetCurrent() {
	CtxSetCurrent(ctx)
}

// Blocks until the device has completed all preceding requested tasks.
func CtxSynchronize() {
	err := Result(C.hipCtxSynchronize())
	if err != SUCCESS {
		panic(err)
	}
}

// Flags for CtxCreate
const (
	// If the number of contexts > number of CPUs, yield to other OS threads when waiting for the GPU, otherwise spin on the processor.
	CTX_SCHED_AUTO = C.hipDeviceScheduleAuto
	// Spin when waiting for results from the GPU.
	CTX_SCHED_SPIN = C.hipDeviceScheduleSpin
	// Yield its thread when waiting for results from the GPU.
	CTX_SCHED_YIELD = C.hipDeviceScheduleYield
	// Block the CPU thread on a synchronization primitive when waiting for the GPU to finish work.
	CTX_BLOCKING_SYNC = C.hipDeviceScheduleBlockingSync
	// Support mapped pinned allocations.
	CTX_MAP_HOST = C.hipDeviceMapHost
	// Do not reduce local memory after resizing local memory for a kernel.
	CTX_LMEM_RESIZE_TO_MAX = C.hipDeviceLmemResizeToMax
)
