//go:build hip

package cu

// This file implements HIP driver device management

//#include <hip/hip_runtime.h>
import "C"

// HIP Device number.
type Device int

// Returns the compute capability of the device.
func DeviceComputeCapability(device Device) (major, minor int) {
	major = device.Attribute(COMPUTE_CAPABILITY_MAJOR)
	minor = device.Attribute(COMPUTE_CAPABILITY_MINOR)
	return
}

// Returns the compute capability of the device.
func (device Device) ComputeCapability() (major, minor int) {
	return DeviceComputeCapability(device)
}

// Returns in a device handle given an ordinal in the range [0, DeviceGetCount()-1].
func DeviceGet(ordinal int) Device {
	var device C.hipDevice_t
	err := Result(C.hipDeviceGet(&device, C.int(ordinal)))
	if err != SUCCESS {
		panic(err)
	}
	return Device(device)
}

// Gets the value of a device attribute.
func DeviceGetAttribute(attrib DeviceAttribute, dev Device) int {
	var attr C.int
	err := Result(C.hipDeviceGetAttribute(&attr, C.hipDeviceAttribute_t(attrib), C.hipDevice_t(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return int(attr)
}

// Gets the value of a device attribute.
func (dev Device) Attribute(attrib DeviceAttribute) int {
	return DeviceGetAttribute(attrib, dev)
}

// Returns the number of devices available for execution.
func DeviceGetCount() int {
	var count C.int
	err := Result(C.hipGetDeviceCount(&count))
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
	err := Result(C.hipDeviceGetName(cstr, C.int(size), C.hipDevice_t(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return C.GoString(cstr)
}

// Gets the name of the device.
func (dev Device) Name() string {
	return DeviceGetName(dev)
}

// Returns the gcnArchName string of the device, e.g. "gfx90a:sramecc+:xnack-".
func DeviceGetArchName(dev Device) string {
	var prop C.hipDeviceProp_t
	err := Result(C.hipGetDeviceProperties(&prop, C.int(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return C.GoString(&prop.gcnArchName[0])
}

// Returns the gcnArchName string of the device.
func (dev Device) ArchName() string {
	return DeviceGetArchName(dev)
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

// Returns the dev's properties.
func DeviceGetProperties(dev Device) (prop DevProp) {
	prop.MaxThreadsPerBlock = dev.Attribute(MAX_THREADS_PER_BLOCK)
	prop.MaxThreadsDim[0] = dev.Attribute(MAX_BLOCK_DIM_X)
	prop.MaxThreadsDim[1] = dev.Attribute(MAX_BLOCK_DIM_Y)
	prop.MaxThreadsDim[2] = dev.Attribute(MAX_BLOCK_DIM_Z)
	prop.MaxGridSize[0] = dev.Attribute(MAX_GRID_DIM_X)
	prop.MaxGridSize[1] = dev.Attribute(MAX_GRID_DIM_Y)
	prop.MaxGridSize[2] = dev.Attribute(MAX_GRID_DIM_Z)
	prop.SharedMemPerBlock = dev.Attribute(MAX_SHARED_MEMORY_PER_BLOCK)
	prop.TotalConstantMemory = dev.Attribute(TOTAL_CONSTANT_MEMORY)
	prop.SIMDWidth = dev.Attribute(WARP_SIZE)
	prop.MemPitch = dev.Attribute(MAX_PITCH)
	prop.RegsPerBlock = dev.Attribute(MAX_REGISTERS_PER_BLOCK)
	prop.ClockRate = dev.Attribute(CLOCK_RATE)
	prop.TextureAlign = dev.Attribute(TEXTURE_ALIGNMENT)
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
	err := Result(C.hipDeviceTotalMem(&bytes, C.hipDevice_t(device)))
	if err != SUCCESS {
		panic(err)
	}
	return int64(bytes)
}

type DeviceAttribute int

const (
	MAX_THREADS_PER_BLOCK          DeviceAttribute = C.hipDeviceAttributeMaxThreadsPerBlock          // Maximum number of threads per block
	MAX_BLOCK_DIM_X                DeviceAttribute = C.hipDeviceAttributeMaxBlockDimX                // Maximum block dimension X
	MAX_BLOCK_DIM_Y                DeviceAttribute = C.hipDeviceAttributeMaxBlockDimY                // Maximum block dimension Y
	MAX_BLOCK_DIM_Z                DeviceAttribute = C.hipDeviceAttributeMaxBlockDimZ                // Maximum block dimension Z
	MAX_GRID_DIM_X                 DeviceAttribute = C.hipDeviceAttributeMaxGridDimX                 // Maximum grid dimension X
	MAX_GRID_DIM_Y                 DeviceAttribute = C.hipDeviceAttributeMaxGridDimY                 // Maximum grid dimension Y
	MAX_GRID_DIM_Z                 DeviceAttribute = C.hipDeviceAttributeMaxGridDimZ                 // Maximum grid dimension Z
	MAX_SHARED_MEMORY_PER_BLOCK    DeviceAttribute = C.hipDeviceAttributeMaxSharedMemoryPerBlock     // Maximum shared memory available per block in bytes
	TOTAL_CONSTANT_MEMORY          DeviceAttribute = C.hipDeviceAttributeTotalConstantMemory         // Memory available on device for __constant__ variables in bytes
	WARP_SIZE                      DeviceAttribute = C.hipDeviceAttributeWarpSize                    // Warp size in threads
	MAX_PITCH                      DeviceAttribute = C.hipDeviceAttributeMaxPitch                    // Maximum pitch in bytes allowed by memory copies
	MAX_REGISTERS_PER_BLOCK        DeviceAttribute = C.hipDeviceAttributeMaxRegistersPerBlock        // Maximum number of 32-bit registers available per block
	CLOCK_RATE                     DeviceAttribute = C.hipDeviceAttributeClockRate                   // Peak clock frequency in kilohertz
	TEXTURE_ALIGNMENT              DeviceAttribute = C.hipDeviceAttributeTextureAlignment            // Alignment requirement for textures
	MULTIPROCESSOR_COUNT           DeviceAttribute = C.hipDeviceAttributeMultiprocessorCount         // Number of multiprocessors on device
	KERNEL_EXEC_TIMEOUT            DeviceAttribute = C.hipDeviceAttributeKernelExecTimeout           // Specifies whether there is a run time limit on kernels
	INTEGRATED                     DeviceAttribute = C.hipDeviceAttributeIntegrated                  // Device is integrated with host memory
	CAN_MAP_HOST_MEMORY            DeviceAttribute = C.hipDeviceAttributeCanMapHostMemory            // Device can map host memory into device address space
	COMPUTE_MODE                   DeviceAttribute = C.hipDeviceAttributeComputeMode                 // Compute mode
	CONCURRENT_KERNELS             DeviceAttribute = C.hipDeviceAttributeConcurrentKernels           // Device can possibly execute multiple kernels concurrently
	ECC_ENABLED                    DeviceAttribute = C.hipDeviceAttributeEccEnabled                  // Device has ECC support enabled
	PCI_BUS_ID                     DeviceAttribute = C.hipDeviceAttributePciBusId                    // PCI bus ID of the device
	PCI_DEVICE_ID                  DeviceAttribute = C.hipDeviceAttributePciDeviceId                 // PCI device ID of the device
	MEMORY_CLOCK_RATE              DeviceAttribute = C.hipDeviceAttributeMemoryClockRate             // Peak memory clock frequency in kilohertz
	GLOBAL_MEMORY_BUS_WIDTH        DeviceAttribute = C.hipDeviceAttributeMemoryBusWidth              // Global memory bus width in bits
	L2_CACHE_SIZE                  DeviceAttribute = C.hipDeviceAttributeL2CacheSize                 // Size of L2 cache in bytes
	MAX_THREADS_PER_MULTIPROCESSOR DeviceAttribute = C.hipDeviceAttributeMaxThreadsPerMultiProcessor // Maximum resident threads per multiprocessor
	ASYNC_ENGINE_COUNT             DeviceAttribute = C.hipDeviceAttributeAsyncEngineCount            // Number of asynchronous engines
	UNIFIED_ADDRESSING             DeviceAttribute = C.hipDeviceAttributeUnifiedAddressing           // Device shares a unified address space with the host
	COMPUTE_CAPABILITY_MAJOR       DeviceAttribute = C.hipDeviceAttributeComputeCapabilityMajor      // Major compute capability version number
	COMPUTE_CAPABILITY_MINOR       DeviceAttribute = C.hipDeviceAttributeComputeCapabilityMinor      // Minor compute capability version number
)
