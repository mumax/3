package main

import (
	. "nimble-cube/nc"
	"runtime"
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
	go func() {
		runtime.LockOSThread()
		kern.Run()
	}()
	go func() {
		runtime.LockOSThread()
		togpu.Run()
	}()
	conv.Run() // once

}
