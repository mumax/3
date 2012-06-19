package main

import (
	"fmt"
	. "nimble-cube/nc"
	"os"
)

func main() {

	MAX_WARPLEN = 4
	n := 4
	InitSize(n, n, n)

	source := new(Source3)
	to := NewToGpu3Seq()
	gpusinkX := new(GpuSink)
	gpusinkY := new(GpuSink)
	gpusinkZ := new(GpuSink)

	Register(source, to, gpusinkX, gpusinkY, gpusinkZ)

	Connect(&to.Input, &source.Output)
	Connect(&gpusinkX.Input, &to.Output[X])
	Connect(&gpusinkY.Input, &to.Output[Y])
	Connect(&gpusinkZ.Input, &to.Output[Z])

	Vet(source, to, gpusinkX, gpusinkY, gpusinkZ)
	WriteGraph("togpuseq3")

	GoRun(to, gpusinkX, gpusinkY, gpusinkZ)
	source.Run(50)

	fmt.Println("NumAlloc:", NumAlloc)
	if NumAlloc > 30 {
		os.Exit(1)
	}
	fmt.Println("NumGpuAlloc:", NumGpuAlloc)
	if NumGpuAlloc > 30 {
		os.Exit(1)
	}
}

type Source3 struct {
	Output [3][]chan<- []float32
}

func (box *Source3) Run(n int) {
	for i := 0; i < n; i++ {
		b := Buffer3()
		for c := range b {
			for i := range b[c] {
				b[c][i] = float32(c)
			}
			Send3(box.Output, b)
		}
	}
}

type GpuSink struct {
	Input <-chan GpuBlock
}

func (box *GpuSink) Run() {
	for {
		RecycleGpu(RecvGpu(box.Input))
	}
}
