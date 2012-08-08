package mm

import (
	. "nimble-cube/nc"
)

type Average3Box struct {
	Input  [3]<-chan Block
	Output [3][]chan<- float64
}

func NewAverage3Box() *Average3Box {
	box := new(Average3Box)
	Register(box)
	return box
}

func (box *Average3Box) Run() {
	for {
		for c := 0; c < 3; c++ {
			sum := 0.0
			for I := 0; I < N(); I += WarpLen() {
				in := Recv(box.Input[c])
				for _, value := range in.List {
					sum += float64(value)
				}
				Recycle(in)
			}
			SendFloat64(box.Output[c], sum/float64(N()))
		}
	}
}
