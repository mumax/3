package main

import (
	. "nimble-cube/nc"
)

func main() {

	MAX_WARPLEN = 256
	InitSize(4, 8, 16)

	kern := NewKernelBox()
	togpu := NewToGpuBox()
	conv := NewConvBox()

	Connect(&togpu.Input, &kern.FFTKernel)
	Connect(&conv.FFTKernel, &togpu.Output)
	WriteGraph("conv")
	GoRun(kern, togpu)
	conv.Run() // once

}
