package cu

// This file implements CUDA driver device management

//#include <cuda.h>
import "C"

import ()

// CUDA Device number.
type Device int

// Returns the compute capability of the device.
func DeviceComputeCapability(device Device) (major, minor int) {
	var maj, min C.int
	err := Result(C.cuDeviceComputeCapability(&maj, &min, C.CUdevice(device)))
	if err != SUCCESS {
		panic(err)
	}
	major = int(maj)
	minor = int(min)
	return
}

// Returns the compute capability of the device.
func (device Device) ComputeCapability() (major, minor int) {
	return DeviceComputeCapability(device)
}

// Returns in a device handle given an ordinal in the range [0, DeviceGetCount()-1].
func DeviceGet(ordinal int) Device {
	var device C.CUdevice
	err := Result(C.cuDeviceGet(&device, C.int(ordinal)))
	if err != SUCCESS {
		panic(err)
	}
	return Device(device)
}

// Gets the value of a device attribute.
func DeviceGetAttribute(attrib DeviceAttribute, dev Device) int {
	var attr C.int
	err := Result(C.cuDeviceGetAttribute(&attr, C.CUdevice_attribute(attrib), C.CUdevice(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return int(attr)
}

// Gets the value of a device attribute.
func (dev Device) Attribute(attrib DeviceAttribute) int {
	return DeviceGetAttribute(attrib, dev)
}

// Returns the number of devices with compute capability greater than or equal to 1.0 that are available for execution.
func DeviceGetCount() int {
	var count C.int
	err := Result(C.cuDeviceGetCount(&count))
	if err != SUCCESS {
		panic(err)
	}
	return int(count)
}

// Gets the name of the device.
func DeviceGetName(dev Device) string {
	size := 256
	buf := make([]byte, size)
	cstr := C.CString(string(buf))
	err := Result(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return C.GoString(cstr)
}

// Gets the name of the device.
func (dev Device) Name() string {
	return DeviceGetName(dev)
}

// Device properties
type DevProp struct {
	MaxThreadsPerBlock  int
	MaxThreadsDim       [3]int
	MaxGridSize         [3]int
	SharedMemPerBlock   int
	TotalConstantMemory int
	SIMDWidth           int
	MemPitch            int
	RegsPerBlock        int
	ClockRate           int
	TextureAlign        int
}

// Returns the device's properties.
func DeviceGetProperties(dev Device) (prop DevProp) {
	var cprop C.CUdevprop
	err := Result(C.cuDeviceGetProperties(&cprop, C.CUdevice(dev)))
	if err != SUCCESS {
		panic(err)
	}
	prop.MaxThreadsPerBlock = int(cprop.maxThreadsPerBlock)
	prop.MaxThreadsDim[0] = int(cprop.maxThreadsDim[0])
	prop.MaxThreadsDim[1] = int(cprop.maxThreadsDim[1])
	prop.MaxThreadsDim[2] = int(cprop.maxThreadsDim[2])
	prop.MaxGridSize[0] = int(cprop.maxGridSize[0])
	prop.MaxGridSize[1] = int(cprop.maxGridSize[1])
	prop.MaxGridSize[2] = int(cprop.maxGridSize[2])
	prop.SharedMemPerBlock = int(cprop.sharedMemPerBlock)
	prop.TotalConstantMemory = int(cprop.totalConstantMemory)
	prop.SIMDWidth = int(cprop.SIMDWidth)
	prop.MemPitch = int(cprop.memPitch)
	prop.RegsPerBlock = int(cprop.regsPerBlock)
	prop.ClockRate = int(cprop.clockRate)
	prop.TextureAlign = int(cprop.textureAlign)
	return
}

// Returns the device's properties.
func (dev Device) Properties() DevProp {
	return DeviceGetProperties(dev)
}

// Returns the total amount of memory available on the device in bytes.
func (device Device) TotalMem() int64 {
	return DeviceTotalMem(device)
}

// Returns the total amount of memory available on the device in bytes.
func DeviceTotalMem(device Device) int64 {
	var bytes C.size_t
	err := Result(C.cuDeviceTotalMem(&bytes, C.CUdevice(device)))
	if err != SUCCESS {
		panic(err)
	}
	return int64(bytes)
}

type DeviceAttribute int

const (
	MAX_THREADS_PER_BLOCK            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK            // Maximum number of threads per block
	MAX_BLOCK_DIM_X                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                  // Maximum block dimension X
	MAX_BLOCK_DIM_Y                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                  // Maximum block dimension Y
	MAX_BLOCK_DIM_Z                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                  // Maximum block dimension Z
	MAX_GRID_DIM_X                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                   // Maximum grid dimension X
	MAX_GRID_DIM_Y                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                   // Maximum grid dimension Y
	MAX_GRID_DIM_Z                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                   // Maximum grid dimension Z
	MAX_SHARED_MEMORY_PER_BLOCK      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK      // Maximum shared memory available per block in bytes
	TOTAL_CONSTANT_MEMORY            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY            // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	WARP_SIZE                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_WARP_SIZE                        // Warp size in threads
	MAX_PITCH                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_PITCH                        // Maximum pitch in bytes allowed by memory copies
	MAX_REGISTERS_PER_BLOCK          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK          // Maximum number of 32-bit registers available per block
	CLOCK_RATE                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CLOCK_RATE                       // Peak clock frequency in kilohertz
	TEXTURE_ALIGNMENT                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                // Alignment requirement for textures
	MULTIPROCESSOR_COUNT             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT             // Number of multiprocessors on device
	KERNEL_EXEC_TIMEOUT              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT              // Specifies whether there is a run time limit on kernels
	INTEGRATED                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_INTEGRATED                       // Device is integrated with host memory
	CAN_MAP_HOST_MEMORY              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY              // Device can map host memory into CUDA address space
	COMPUTE_MODE                     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                     // Compute mode (See ::CUcomputemode for details)
	MAXIMUM_TEXTURE1D_WIDTH          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH          // Maximum 1D texture width
	MAXIMUM_TEXTURE2D_WIDTH          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH          // Maximum 2D texture width
	MAXIMUM_TEXTURE2D_HEIGHT         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT         // Maximum 2D texture height
	MAXIMUM_TEXTURE3D_WIDTH          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH          // Maximum 3D texture width
	MAXIMUM_TEXTURE3D_HEIGHT         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT         // Maximum 3D texture height
	MAXIMUM_TEXTURE3D_DEPTH          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH          // Maximum 3D texture depth
	MAXIMUM_TEXTURE2D_LAYERED_WIDTH  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH  // Maximum 2D layered texture width
	MAXIMUM_TEXTURE2D_LAYERED_HEIGHT DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT // Maximum 2D layered texture height
	MAXIMUM_TEXTURE2D_LAYERED_LAYERS DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS // Maximum layers in a 2D layered texture
	SURFACE_ALIGNMENT                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                // Alignment requirement for surfaces
	CONCURRENT_KERNELS               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS               // Device can possibly execute multiple kernels concurrently
	ECC_ENABLED                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ECC_ENABLED                      // Device has ECC support enabled
	PCI_BUS_ID                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                       // PCI bus ID of the device
	PCI_DEVICE_ID                    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                    // PCI device ID of the device
	TCC_DRIVER                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TCC_DRIVER                       // Device is using TCC driver model
	MEMORY_CLOCK_RATE                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                // Peak memory clock frequency in kilohertz
	GLOBAL_MEMORY_BUS_WIDTH          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH          // Global memory bus width in bits
	L2_CACHE_SIZE                    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                    // Size of L2 cache in bytes
	MAX_THREADS_PER_MULTIPROCESSOR   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR   // Maximum resident threads per multiprocessor
	ASYNC_ENGINE_COUNT               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT               // Number of asynchronous engines
	UNIFIED_ADDRESSING               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING               // Device uses shares a unified address space with the host
	MAXIMUM_TEXTURE1D_LAYERED_WIDTH  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH  // Maximum 1D layered texture width
	MAXIMUM_TEXTURE1D_LAYERED_LAYERS DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS // Maximum layers in a 1D layered texture
)
