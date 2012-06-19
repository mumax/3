package mm

import (
	. "nimble-cube/nc"
)

type AverageBox struct {
	Input  <-chan Block
	Output []chan<- float64
}

func (box *AverageBox) Run() {
	for {
		sum := 0.0
		for I := 0; I < N(); I += WarpLen() {
			in := Recv(box.Input)
			for _, value := range in.List {
				sum += float64(value)
			}
			Recycle(in)
		}
		SendFloat64(box.Output, sum/float64(N()))
	}
}
