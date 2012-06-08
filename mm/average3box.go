package mm

import (
	. "nimble-cube/nc"
)

type Average3Box struct {
	Input  [3]<-chan []float32
	Output [3][]chan<- float64
}

func NewAverage3Box(quant string) *Average3Box {
	avg := new(Average3Box)
	ConnectToQuant(avg, &avg.Input, quant)
	RegisterChannel(avg, &avg.Output, "<"+quant+">")
	return avg
}

func (box *Average3Box) Run() {
	for {
		for c := 0; c < 3; c++ {
			sum := 0.0
			for I := 0; I < N(); I += WarpLen() {
				in := Recv(box.Input[c])
				for _, value := range in {
					sum += float64(value)
				}
			}
			SendFloat64(box.Output[c], sum/float64(N()))
		}
	}
}
