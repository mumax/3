package cu

import (
	"fmt"
	"testing"
)

func TestDevice(t *testing.T) {
	fmt.Println("DeviceGetCount:", DeviceGetCount())
	for i := 0; i < DeviceGetCount(); i++ {
		fmt.Println("DeviceGet", i)
		dev := DeviceGet(i)
		major, minor := dev.ComputeCapability()
		fmt.Println("Name: ", dev.Name())
		fmt.Println("ComputeCapability: ", major, minor)
		fmt.Println("TotalMem: ", dev.TotalMem())

		fmt.Println("ATTRIBUTE_MAX_THREADS_PER_BLOCK           :", dev.Attribute(MAX_THREADS_PER_BLOCK))
		fmt.Println("ATTRIBUTE_MAX_BLOCK_DIM_X                 :", dev.Attribute(MAX_BLOCK_DIM_X))
		fmt.Println("ATTRIBUTE_MAX_BLOCK_DIM_Y                 :", dev.Attribute(MAX_BLOCK_DIM_Y))
		fmt.Println("ATTRIBUTE_MAX_BLOCK_DIM_Z                 :", dev.Attribute(MAX_BLOCK_DIM_Z))
		fmt.Println("ATTRIBUTE_MAX_GRID_DIM_X                  :", dev.Attribute(MAX_GRID_DIM_X))
		fmt.Println("ATTRIBUTE_MAX_GRID_DIM_Y                  :", dev.Attribute(MAX_GRID_DIM_Y))
		fmt.Println("ATTRIBUTE_MAX_GRID_DIM_Z                  :", dev.Attribute(MAX_GRID_DIM_Z))
		fmt.Println("ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK     :", dev.Attribute(MAX_SHARED_MEMORY_PER_BLOCK))
		fmt.Println("ATTRIBUTE_TOTAL_CONSTANT_MEMORY           :", dev.Attribute(TOTAL_CONSTANT_MEMORY))
		fmt.Println("ATTRIBUTE_WARP_SIZE                       :", dev.Attribute(WARP_SIZE))
		fmt.Println("ATTRIBUTE_MAX_PITCH                       :", dev.Attribute(MAX_PITCH))
		fmt.Println("ATTRIBUTE_MAX_REGISTERS_PER_BLOCK         :", dev.Attribute(MAX_REGISTERS_PER_BLOCK))
		fmt.Println("ATTRIBUTE_CLOCK_RATE                      :", dev.Attribute(CLOCK_RATE))
		fmt.Println("ATTRIBUTE_TEXTURE_ALIGNMENT               :", dev.Attribute(TEXTURE_ALIGNMENT))
		fmt.Println("ATTRIBUTE_MULTIPROCESSOR_COUNT            :", dev.Attribute(MULTIPROCESSOR_COUNT))
		fmt.Println("ATTRIBUTE_KERNEL_EXEC_TIMEOUT             :", dev.Attribute(KERNEL_EXEC_TIMEOUT))
		fmt.Println("ATTRIBUTE_INTEGRATED                      :", dev.Attribute(INTEGRATED))
		fmt.Println("ATTRIBUTE_CAN_MAP_HOST_MEMORY             :", dev.Attribute(CAN_MAP_HOST_MEMORY))
		fmt.Println("ATTRIBUTE_COMPUTE_MODE                    :", dev.Attribute(COMPUTE_MODE))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH         :", dev.Attribute(MAXIMUM_TEXTURE1D_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH         :", dev.Attribute(MAXIMUM_TEXTURE2D_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT        :", dev.Attribute(MAXIMUM_TEXTURE2D_HEIGHT))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH         :", dev.Attribute(MAXIMUM_TEXTURE3D_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT        :", dev.Attribute(MAXIMUM_TEXTURE3D_HEIGHT))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH         :", dev.Attribute(MAXIMUM_TEXTURE3D_DEPTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH :", dev.Attribute(MAXIMUM_TEXTURE2D_LAYERED_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:", dev.Attribute(MAXIMUM_TEXTURE2D_LAYERED_HEIGHT))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:", dev.Attribute(MAXIMUM_TEXTURE2D_LAYERED_LAYERS))
		fmt.Println("ATTRIBUTE_SURFACE_ALIGNMENT               :", dev.Attribute(SURFACE_ALIGNMENT))
		fmt.Println("ATTRIBUTE_CONCURRENT_KERNELS              :", dev.Attribute(CONCURRENT_KERNELS))
		fmt.Println("ATTRIBUTE_ECC_ENABLED                     :", dev.Attribute(ECC_ENABLED))
		fmt.Println("ATTRIBUTE_PCI_BUS_ID                      :", dev.Attribute(PCI_BUS_ID))
		fmt.Println("ATTRIBUTE_PCI_DEVICE_ID                   :", dev.Attribute(PCI_DEVICE_ID))
		fmt.Println("ATTRIBUTE_TCC_DRIVER                      :", dev.Attribute(TCC_DRIVER))
		fmt.Println("ATTRIBUTE_MEMORY_CLOCK_RATE               :", dev.Attribute(MEMORY_CLOCK_RATE))
		fmt.Println("ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH         :", dev.Attribute(GLOBAL_MEMORY_BUS_WIDTH))
		fmt.Println("ATTRIBUTE_L2_CACHE_SIZE                   :", dev.Attribute(L2_CACHE_SIZE))
		fmt.Println("ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR  :", dev.Attribute(MAX_THREADS_PER_MULTIPROCESSOR))
		fmt.Println("ATTRIBUTE_ASYNC_ENGINE_COUNT              :", dev.Attribute(ASYNC_ENGINE_COUNT))
		fmt.Println("ATTRIBUTE_UNIFIED_ADDRESSING              :", dev.Attribute(UNIFIED_ADDRESSING))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH :", dev.Attribute(MAXIMUM_TEXTURE1D_LAYERED_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:", dev.Attribute(MAXIMUM_TEXTURE1D_LAYERED_LAYERS))

		fmt.Printf("Properties:%#v\n", dev.Properties())
	}
}
