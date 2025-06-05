package cu

// This file implements CUDA driver context management

//#include <cuda.h>
import "C"
import "unsafe"

// CUDA context.
type Context uintptr

// Create a CUDA context.
func CtxCreate(flags uint, dev Device) Context {
	var ctx C.CUcontext
	err := Result(C.cuCtxCreate(&ctx, C.uint(flags), C.CUdevice(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return Context(uintptr(unsafe.Pointer(ctx)))
}

// Destroys the CUDA context specified by ctx. If the context usage count is not equal to 1, or the context is current to any CPU thread other than the current one, this function fails. Floating contexts (detached from a CPU thread via cuCtxPopCurrent()) may be destroyed by this function.
func CtxDestroy(ctx *Context) {
	err := Result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(uintptr(*ctx)))))
	*ctx = 0
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys the CUDA context.
func (ctx *Context) Destroy() {
	CtxDestroy(ctx)
}

// Returns the API version to create the context.
func CtxGetApiVersion(ctx Context) (version int) {
	var cversion C.uint
	err := Result(C.cuCtxGetApiVersion(C.CUcontext(unsafe.Pointer(uintptr(ctx))), &cversion))
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
	var ctx C.CUcontext
	err := Result(C.cuCtxGetCurrent(&ctx))
	if err != SUCCESS {
		panic(err)
	}
	return Context(uintptr(unsafe.Pointer(ctx)))
}

// Returns the ordinal of the current context's device.
func CtxGetDevice() Device {
	var dev C.CUdevice
	err := Result(C.cuCtxGetDevice(&dev))
	if err != SUCCESS {
		panic(err)
	}
	return Device(dev)
}

// Sets the current active context.
func CtxSetCurrent(ctx Context) {
	err := Result(C.cuCtxSetCurrent(C.CUcontext(unsafe.Pointer(uintptr(ctx)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Sets the current active context.
func (ctx Context) SetCurrent() {
	CtxSetCurrent(ctx)
}

// Blocks until the device has completed all preceding requested tasks, if the context was created with the CU_CTX_SCHED_BLOCKING_SYNC flag.
func CtxSynchronize() {
	err := Result(C.cuCtxSynchronize())
	if err != SUCCESS {
		panic(err)
	}
}

// Flags for CtxCreate
const (
	// If  the number of contexts > number of CPUs, yield to other OS threads when waiting for the GPU, otherwise CUDA spin on the processor.
	CTX_SCHED_AUTO = C.CU_CTX_SCHED_AUTO
	// Spin when waiting for results from the GPU.
	CTX_SCHED_SPIN = C.CU_CTX_SCHED_SPIN
	// Yield its thread when waiting for results from the GPU.
	CTX_SCHED_YIELD = C.CU_CTX_SCHED_YIELD
	// Bock the CPU thread on a synchronization primitive when waiting for the GPU to finish work.
	CTX_BLOCKING_SYNC
	// Support mapped pinned allocations. This flag must be set in order to allocate pinned host memory that is accessible to the GPU.
	CTX_MAP_HOST = C.CU_CTX_MAP_HOST
	//Do not reduce local memory after resizing local memory for a kernel.
	CTX_LMEM_RESIZE_TO_MAX = C.CU_CTX_LMEM_RESIZE_TO_MAX
)
