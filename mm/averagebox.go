package mm

import (
	. "nimble-cube/nc"
)

type AverageBox struct {
	Input  <-chan []float32
	Output []chan<- float64
}

func (box *AverageBox) Run() {
	for {
		sum := 0.0
		for I := 0; I < N(); I += WarpLen() {
			in := <-box.Input
			for _, value := range in {
				sum += float64(value)
			}
		}
		SendFloat64(box.Output, sum/float64(N()))
	}
}
