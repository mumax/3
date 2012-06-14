package main

import (
	"flag"
	"fmt"
	. "nimble-cube/nc"
	"strconv"
	"time"
)

func main() {

	n, err := strconv.Atoi(flag.Arg(0))
	PanicErr(err)
	InitSize(n, n, n)

	source := new(Source)
	to := NewToGpuBox()
	from := NewFromGpuBox()
	sink := new(Sink)

	Connect(&to.Input, &source.Output)
	Connect(&from.Input, &to.Output)
	Connect(&sink.Input, &from.Output)

	GoRun(source, to, from)

	Register(sink)
	WriteGraph("pcie")

	const runs = 100
	start := time.Now()
	sink.Run(runs)
	duration := time.Now().Sub(start)
	fmt.Println("# N	warp  MB  s   MB/s")
	MB := float64((8 * int64(N()) * runs) / 1e6)
	seconds := float64(duration) / 1e9
	fmt.Println(N(), "\t", WarpLen(), "\t", MB, seconds, float64(MB)/seconds)
}

type Source struct {
	Output []chan<- []float32
}

func (box *Source) Run() {
	cnst := make([]float32, WarpLen())
	for {
		Send(box.Output, cnst)
	}
}

type Sink struct {
	Input <-chan []float32
}

func (box *Sink) Run(n int) {
	for step := 0; step < n; step++ {
		//Debug("step", step)
		for slice := 0; slice < NumWarp(); slice++ {
			in := Recv(box.Input)
			Recycle(in)
		}
	}
}
