//go:build hip

package cu

// This file implements HIP memory management on the driver level

//#include <hip/hip_runtime.h>
import "C"

import (
	"fmt"
	"unsafe"
)

type DevicePtr uintptr

func (p DevicePtr) hip() C.hipDeviceptr_t {
	return C.hipDeviceptr_t(unsafe.Pointer(uintptr(p)))
}

// Allocates a number of bytes of device memory.
func MemAlloc(bytes int64) DevicePtr {
	var devptr unsafe.Pointer
	err := Result(C.hipMalloc(&devptr, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
	return DevicePtr(uintptr(devptr))
}

// Frees device memory allocated by MemAlloc().
// It is safe to double-free.
func MemFree(p DevicePtr) {
	if p == DevicePtr(uintptr(0)) {
		return // Allready freed
	}
	err := Result(C.hipFree(unsafe.Pointer(uintptr(p))))
	if err != SUCCESS {
		panic(err)
	}
}

// Frees device memory allocated by MemAlloc().
// Overwrites the pointer with NULL.
// It is safe to double-free.
func (ptr DevicePtr) Free() {
	MemFree(ptr)
}

// Copies a number of bytes on the current device.
func Memcpy(dst, src DevicePtr, bytes int64) {
	err := Result(C.hipMemcpyDtoD(dst.hip(), src.hip(), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes on the current device.
func MemcpyAsync(dst, src DevicePtr, bytes int64, stream Stream) {
	err := Result(C.hipMemcpyDtoDAsync(dst.hip(), src.hip(), C.size_t(bytes), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from device to device.
func MemcpyDtoD(dst, src DevicePtr, bytes int64) {
	err := Result(C.hipMemcpyDtoD(dst.hip(), src.hip(), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes from device to device.
func MemcpyDtoDAsync(dst, src DevicePtr, bytes int64, stream Stream) {
	err := Result(C.hipMemcpyDtoDAsync(dst.hip(), src.hip(), C.size_t(bytes), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from host to device.
func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, bytes int64) {
	err := Result(C.hipMemcpyHtoD(dst.hip(), src, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes from host to device.
// The host memory must be page-locked (see MemHostRegister)
func MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, bytes int64, stream Stream) {
	err := Result(C.hipMemcpyHtoDAsync(dst.hip(), src, C.size_t(bytes), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from device to host.
func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, bytes int64) {
	err := Result(C.hipMemcpyDtoH(dst, src.hip(), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes device host to host.
// The host memory must be page-locked (see MemHostRegister)
func MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, bytes int64, stream Stream) {
	err := Result(C.hipMemcpyDtoHAsync(dst, src.hip(), C.size_t(bytes), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies device memory from one device to another.
func MemcpyPeer(dst DevicePtr, dstDev Device, src DevicePtr, srcDev Device, bytes int64) {
	err := Result(C.hipMemcpyPeer(unsafe.Pointer(uintptr(dst)), C.int(dstDev), unsafe.Pointer(uintptr(src)), C.int(srcDev), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies device memory from one device to another.
func MemcpyPeerAsync(dst DevicePtr, dstDev Device, src DevicePtr, srcDev Device, bytes int64, stream Stream) {
	err := Result(C.hipMemcpyPeerAsync(unsafe.Pointer(uintptr(dst)), C.int(dstDev), unsafe.Pointer(uintptr(src)), C.int(srcDev), C.size_t(bytes), C.hipStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Returns the base address and size of the allocation (by MemAlloc) that contains the input pointer ptr.
func MemGetAddressRange(ptr DevicePtr) (bytes int64, base DevicePtr) {
	var cbytes C.size_t
	var cptr C.hipDeviceptr_t
	err := Result(C.hipMemGetAddressRange(&cptr, &cbytes, ptr.hip()))
	if err != SUCCESS {
		panic(err)
	}
	bytes = int64(cbytes)
	base = DevicePtr(uintptr(unsafe.Pointer(cptr)))
	return
}

// Returns the base address and size of the allocation (by MemAlloc) that contains the input pointer ptr.
func (ptr DevicePtr) GetAddressRange() (bytes int64, base DevicePtr) {
	return MemGetAddressRange(ptr)
}

// Returns the size of the allocation (by MemAlloc) that contains the input pointer ptr.
func (ptr DevicePtr) Bytes() (bytes int64) {
	bytes, _ = MemGetAddressRange(ptr)
	return
}

// Returns the free and total amount of memory in the current Context (in bytes).
func MemGetInfo() (free, total int64) {
	var cfree, ctotal C.size_t
	err := Result(C.hipMemGetInfo(&cfree, &ctotal))
	if err != SUCCESS {
		panic(err)
	}
	free = int64(cfree)
	total = int64(ctotal)
	return
}

func MemAllocHost(bytes int64) unsafe.Pointer {
	var p unsafe.Pointer
	err := Result(C.hipMemAllocHost(&p, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
	return p
}

func MemFreeHost(ptr unsafe.Pointer) {
	err := Result(C.hipHostFree(ptr))
	if err != SUCCESS {
		panic(err)
	}
}

func (p DevicePtr) String() string {
	return fmt.Sprint(unsafe.Pointer(uintptr(p)))
}

// Type size in bytes
const (
	SIZEOF_FLOAT32    = 4
	SIZEOF_FLOAT64    = 8
	SIZEOF_COMPLEX64  = 8
	SIZEOF_COMPLEX128 = 16
)

// Physical memory type of device pointer.
type MemoryType uint

const (
	MemoryTypeHost    MemoryType = C.hipMemoryTypeHost
	MemoryTypeDevice  MemoryType = C.hipMemoryTypeDevice
	MemoryTypeArray   MemoryType = C.hipMemoryTypeArray
	MemoryTypeUnified MemoryType = C.hipMemoryTypeUnified
)

var memorytype = map[MemoryType]string{
	MemoryTypeHost:    "MemoryTypeHost",
	MemoryTypeDevice:  "MemoryTypeDevice",
	MemoryTypeArray:   "MemoryTypeArray",
	MemoryTypeUnified: "MemoryTypeUnified"}

func (t MemoryType) String() string {
	if s, ok := memorytype[t]; ok {
		return s
	}
	return "MemoryTypeUnknown"
}

// Returns the physical memory type that ptr addresses.
func PointerGetAttributeMemoryType(ptr DevicePtr) (t MemoryType, err Result) {
	var typ uint64 // foresee enough memory just to be safe
	err = Result(C.hipPointerGetAttribute(unsafe.Pointer(&typ),
		C.HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr.hip()))
	return MemoryType(uint(typ)), err
}

// Returns the physical memory type that ptr addresses.
func (ptr DevicePtr) MemoryType() MemoryType {
	t, err := PointerGetAttributeMemoryType(ptr)
	if err != SUCCESS {
		panic(err)
	}
	return t
}
