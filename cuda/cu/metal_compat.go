//go:build darwin && arm64

package cu

// This file preserves the small CUDA-driver surface still used by MuMax3's
// backend-neutral Go code. On Apple Silicon it delegates memory and stream
// ordering to cuda/metal. Kernel modules and launches are intentionally not
// emulated here: generated Darwin wrappers call the typed Metal API directly.

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
	"github.com/mumax/3/cuda/metal/kernels"
)

const CUDA_VERSION = 0

const (
	SIZEOF_FLOAT32    = 4
	SIZEOF_FLOAT64    = 8
	SIZEOF_COMPLEX64  = 8
	SIZEOF_COMPLEX128 = 16
)

// Result retains the CUDA package's panic/comparison contract for the handful
// of legacy call sites that distinguish allocation and readiness failures.
type Result int

const (
	SUCCESS                 Result = 0
	ERROR_INVALID_VALUE     Result = 1
	ERROR_OUT_OF_MEMORY     Result = 2
	ERROR_NOT_INITIALIZED   Result = 3
	ERROR_NO_DEVICE         Result = 100
	ERROR_INVALID_DEVICE    Result = 101
	ERROR_NO_BINARY_FOR_GPU Result = 209
	ERROR_NOT_READY         Result = 600
	ERROR_NOT_SUPPORTED     Result = 801
	ERROR_UNKNOWN           Result = 999
)

func (result Result) String() string {
	switch result {
	case SUCCESS:
		return "SUCCESS"
	case ERROR_INVALID_VALUE:
		return "ERROR_INVALID_VALUE"
	case ERROR_OUT_OF_MEMORY:
		return "ERROR_OUT_OF_MEMORY"
	case ERROR_NOT_INITIALIZED:
		return "ERROR_NOT_INITIALIZED"
	case ERROR_NO_DEVICE:
		return "ERROR_NO_DEVICE"
	case ERROR_INVALID_DEVICE:
		return "ERROR_INVALID_DEVICE"
	case ERROR_NO_BINARY_FOR_GPU:
		return "ERROR_NO_BINARY_FOR_GPU"
	case ERROR_NOT_READY:
		return "ERROR_NOT_READY"
	case ERROR_NOT_SUPPORTED:
		return "ERROR_NOT_SUPPORTED"
	default:
		return fmt.Sprintf("Metal runtime error %d", result)
	}
}

func panicMetal(err error) {
	if err == nil {
		return
	}
	var runtimeError *metal.RuntimeError
	if errors.As(err, &runtimeError) {
		switch runtimeError.Code {
		case 1:
			panic(ERROR_NO_DEVICE)
		case 2:
			panic(ERROR_INVALID_VALUE)
		case 3:
			panic(ERROR_OUT_OF_MEMORY)
		}
	}
	panic(err)
}

// Init initializes the process-wide Metal runtime. CUDA's flags argument is
// retained for source compatibility and must remain zero.
var (
	initializeMetalOnce sync.Once
	initializeMetalErr  error
)

func Init(flags int) {
	if flags != 0 {
		panic(ERROR_INVALID_VALUE)
	}
	initializeMetalOnce.Do(func() {
		if err := metal.Initialize(); err != nil {
			initializeMetalErr = err
			return
		}
		initializeMetalErr = metal.RegisterSource(kernels.Source)
	})
	panicMetal(initializeMetalErr)
}

func Version() int {
	return 300
}

// Context is a compatibility token. Metal command queues are process-wide and
// do not have CUDA's thread-current context semantics.
type Context uintptr

const (
	CTX_SCHED_AUTO         = 0
	CTX_SCHED_SPIN         = 1
	CTX_SCHED_YIELD        = 2
	CTX_BLOCKING_SYNC      = 4
	CTX_MAP_HOST           = 8
	CTX_LMEM_RESIZE_TO_MAX = 16
)

func CtxCreate(flags uint, dev Device) Context {
	if dev != 0 {
		panic(ERROR_INVALID_DEVICE)
	}
	panicMetal(metal.Initialize())
	return Context(1)
}

func CtxDestroy(context *Context) {
	if context != nil {
		*context = 0
	}
}

func (context *Context) Destroy() {
	CtxDestroy(context)
}

func CtxGetApiVersion(context Context) int {
	return Version()
}

func (context Context) ApiVersion() int {
	return CtxGetApiVersion(context)
}

func CtxGetCurrent() Context {
	if metal.Available() {
		return Context(1)
	}
	return 0
}

func CtxGetDevice() Device {
	return Device(0)
}

func CtxSetCurrent(context Context) {
	if context != 0 {
		panicMetal(metal.Initialize())
	}
}

func (context Context) SetCurrent() {
	CtxSetCurrent(context)
}

func CtxSynchronize() {
	panicMetal(metal.Sync())
}

func CtxEnablePeerAccess(peer Context) {}

func (peer Context) EnablePeerAccess() {}

func CtxDisablePeerAccess(peer Context) {}

func (peer Context) DisablePeerAccess() {}

// Device is always zero: Apple exposes the selected system-default GPU.
type Device int

func DeviceGet(ordinal int) Device {
	if ordinal != 0 || !metal.Available() {
		panic(ERROR_INVALID_DEVICE)
	}
	return Device(0)
}

func DeviceGetCount() int {
	if metal.Available() {
		return 1
	}
	return 0
}

func DeviceGetName(device Device) string {
	if device != 0 {
		panic(ERROR_INVALID_DEVICE)
	}
	info, err := metal.Info()
	panicMetal(err)
	return info.Name
}

func (device Device) Name() string {
	return DeviceGetName(device)
}

func DeviceTotalMem(device Device) int64 {
	if device != 0 {
		panic(ERROR_INVALID_DEVICE)
	}
	info, err := metal.Info()
	panicMetal(err)
	return int64(info.RecommendedMaxWorkingSetSize)
}

func (device Device) TotalMem() int64 {
	return DeviceTotalMem(device)
}

func DeviceComputeCapability(device Device) (int, int) {
	return 0, 0
}

func (device Device) ComputeCapability() (int, int) {
	return DeviceComputeCapability(device)
}

func DeviceCanAccessPeer(device, peer Device) bool {
	return false
}

func (device Device) CanAccessPeer(peer Device) bool {
	return false
}

type DeviceAttribute int

const (
	MAX_THREADS_PER_BLOCK DeviceAttribute = iota
	MAX_BLOCK_DIM_X
	MAX_BLOCK_DIM_Y
	MAX_BLOCK_DIM_Z
	MAX_GRID_DIM_X
	MAX_GRID_DIM_Y
	MAX_GRID_DIM_Z
	WARP_SIZE
	INTEGRATED
	UNIFIED_ADDRESSING
)

func DeviceGetAttribute(attribute DeviceAttribute, device Device) int {
	if device != 0 {
		panic(ERROR_INVALID_DEVICE)
	}
	info, err := metal.Info()
	panicMetal(err)
	switch attribute {
	case MAX_THREADS_PER_BLOCK:
		return int(info.MaxThreadsPerThreadgroup[0])
	case MAX_BLOCK_DIM_X:
		return int(info.MaxThreadsPerThreadgroup[0])
	case MAX_BLOCK_DIM_Y:
		return int(info.MaxThreadsPerThreadgroup[1])
	case MAX_BLOCK_DIM_Z:
		return int(info.MaxThreadsPerThreadgroup[2])
	case WARP_SIZE:
		return 32
	case INTEGRATED, UNIFIED_ADDRESSING:
		return 1
	default:
		return 0
	}
}

func (device Device) Attribute(attribute DeviceAttribute) int {
	return DeviceGetAttribute(attribute, device)
}

// Stream is a compatibility token for Metal's single ordered command queue.
type Stream uintptr

func StreamCreate() Stream {
	panicMetal(metal.Initialize())
	return Stream(0)
}

func (stream *Stream) Destroy() {
	if stream != nil {
		*stream = 0
	}
}

func StreamDestroy(stream *Stream) {
	stream.Destroy()
}

func (stream Stream) Synchronize() {
	panicMetal(metal.Sync())
}

func StreamSynchronize(stream Stream) {
	stream.Synchronize()
}

func (stream Stream) Query() Result {
	return SUCCESS
}

func StreamQuery(stream Stream) Result {
	return stream.Query()
}

// DevicePtr is the CPU-visible address of a shared MTLBuffer allocation.
// Interior pointer arithmetic remains valid.
type DevicePtr uintptr

func MemAlloc(bytes int64) DevicePtr {
	pointer, err := metal.Alloc(bytes)
	panicMetal(err)
	return DevicePtr(uintptr(pointer))
}

func MemFree(pointer DevicePtr) {
	panicMetal(metal.Free(unsafe.Pointer(uintptr(pointer))))
}

func (pointer DevicePtr) Free() {
	MemFree(pointer)
}

func MemGetAddressRange(pointer DevicePtr) (int64, DevicePtr) {
	base, bytes, err := metal.AllocationRange(
		unsafe.Pointer(uintptr(pointer)),
	)
	panicMetal(err)
	return bytes, DevicePtr(uintptr(base))
}

func (pointer DevicePtr) GetAddressRange() (int64, DevicePtr) {
	return MemGetAddressRange(pointer)
}

func (pointer DevicePtr) Bytes() int64 {
	bytes, _ := MemGetAddressRange(pointer)
	return bytes
}

func MemGetInfo() (int64, int64) {
	info, err := metal.Info()
	panicMetal(err)
	total := info.RecommendedMaxWorkingSetSize
	used := info.CurrentAllocatedSize
	if used > total {
		used = total
	}
	return int64(total - used), int64(total)
}

func Memcpy(dst, src DevicePtr, bytes int64) {
	panicMetal(metal.Copy(
		unsafe.Pointer(uintptr(dst)),
		unsafe.Pointer(uintptr(src)),
		bytes,
	))
}

func MemcpyAsync(dst, src DevicePtr, bytes int64, stream Stream) {
	Memcpy(dst, src, bytes)
}

func MemcpyDtoD(dst, src DevicePtr, bytes int64) {
	Memcpy(dst, src, bytes)
}

func MemcpyDtoDAsync(dst, src DevicePtr, bytes int64, stream Stream) {
	Memcpy(dst, src, bytes)
}

func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, bytes int64) {
	panicMetal(metal.CopyToDevice(
		unsafe.Pointer(uintptr(dst)),
		src,
		bytes,
	))
}

func MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, bytes int64, stream Stream) {
	MemcpyHtoD(dst, src, bytes)
}

func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, bytes int64) {
	panicMetal(metal.CopyToHost(
		dst,
		unsafe.Pointer(uintptr(src)),
		bytes,
	))
}

func MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, bytes int64, stream Stream) {
	MemcpyDtoH(dst, src, bytes)
}

func MemcpyPeer(dst DevicePtr, dstContext Context, src DevicePtr, srcContext Context, bytes int64) {
	Memcpy(dst, src, bytes)
}

func MemcpyPeerAsync(dst DevicePtr, dstContext Context, src DevicePtr, srcContext Context, bytes int64, stream Stream) {
	Memcpy(dst, src, bytes)
}

func MemAllocHost(bytes int64) unsafe.Pointer {
	pointer, err := metal.Alloc(bytes)
	panicMetal(err)
	return pointer
}

func MemFreeHost(pointer unsafe.Pointer) {
	panicMetal(metal.Free(pointer))
}

type MemHostRegisterFlag int

const (
	MEMHOSTREGISTER_PORTABLE  MemHostRegisterFlag = 1
	MEMHOSTREGISTER_DEVICEMAP MemHostRegisterFlag = 2
)

func MemsetD32(pointer DevicePtr, value uint32, count int64) {
	panicMetal(metal.FillUint32(
		unsafe.Pointer(uintptr(pointer)),
		value,
		count,
	))
}

func MemsetD32Async(pointer DevicePtr, value uint32, count int64, stream Stream) {
	MemsetD32(pointer, value, count)
}

func MemsetD8(pointer DevicePtr, value uint8, count int64) {
	panicMetal(metal.Fill(
		unsafe.Pointer(uintptr(pointer)),
		value,
		count,
	))
}

func MemsetD8Async(pointer DevicePtr, value uint8, count int64, stream Stream) {
	MemsetD8(pointer, value, count)
}

func (pointer DevicePtr) String() string {
	return fmt.Sprint(unsafe.Pointer(uintptr(pointer)))
}
