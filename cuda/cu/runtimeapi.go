package cu

// This file implements parts of the CUDA runtime api instead of the driver
// api the rest of this package uses.
// It might be useful to move this to a seperate package at some point.

//#include <cuda_runtime.h>
import "C"
import "unsafe"

// Set the device as current.
func SetDevice(device Device) {
	err := Result(C.cudaSetDevice(C.int(device)))
	if err != SUCCESS {
		panic(err)
	}
}

// Reset the state of the current device.
func DeviceReset() {
	err := Result(C.cudaDeviceReset())
	if err != SUCCESS {
		panic(err)
	}
}

// Set CUDA device flags.
func SetDeviceFlags(flags uint) {
	err := Result(C.cudaSetDeviceFlags(C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}

//Flags for SetDeviceFlasgs
const (
	// The default, decides to yield or not based on active CUDA threads and processors.
	DeviceAuto = C.cudaDeviceScheduleAuto
	// Actively spin while waiting for device.
	DeviceSpin = C.cudaDeviceScheduleSpin
	// Yield when waiting.
	DeviceYield = C.cudaDeviceScheduleYield
	// ScheduleBlockingSync block CPU on sync.
	DeviceScheduleBlockingSync = C.cudaDeviceScheduleBlockingSync
	// ScheduleBlockingSync block CPU on sync.  Deprecated since cuda 4.0
	DeviceBlockingSync = C.cudaDeviceBlockingSync
	// For use with pinned host memory
	DeviceMapHost = C.cudaDeviceMapHost
	// Do not reduce local memory to try and prevent thrashing
	DeviceLmemResizeToMax = C.cudaDeviceLmemResizeToMax
)

func Malloc(bytes int64) DevicePtr {
	var devptr unsafe.Pointer
	err := Result(C.cudaMalloc(&devptr, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
	return DevicePtr(devptr)
}

func MallocHost(bytes int64) unsafe.Pointer {
	var p unsafe.Pointer
	err := Result(C.cudaMallocHost(&p, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
	return p
}

func FreeHost(ptr unsafe.Pointer) {
	err := Result(C.cudaFreeHost(ptr))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes in the direction specified by flags
func MemCpy(dst, src unsafe.Pointer, bytes int64, flags uint) {
	err := Result(C.cudaMemcpy(dst, src, C.size_t(bytes), uint32(flags)))
	if err != SUCCESS {
		panic(err)
	}
}

//Flags for memory copy types
const (
	// Host to Host
	HtoH = C.cudaMemcpyHostToHost
	// Host to Device
	HtoD = C.cudaMemcpyHostToDevice
	// Device to Host
	DtoH = C.cudaMemcpyDeviceToHost
	// Device to Device
	DtoD = C.cudaMemcpyDeviceToDevice
	// Default, unified virtual address space
	Virt = C.cudaMemcpyDefault
)
