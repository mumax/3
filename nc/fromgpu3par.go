package nc

import (
	"github.com/barnex/cuda4/cu"
)

// Copies a 3-vector from GPU,
// interleaving the components as much as possible. 
// I.e.: Tries to copy X,Y,Z X,Y,Z X,Y,Z, ...
// However, if a component is ready to be sent it is skipped,
// so the box will not idle.
type FromGpu3Par struct {
	Input  [3]chan<- GpuFloats
	Output [3][]<-chan []float32
	stream cu.Stream
}

func NewFromGpu3Par() *FromGpu3Par {
	box := new(FromGpu3Par)
	box.stream = cu.StreamCreate()
	Register(box)
	return box
}

func (box *FromGpu3Par) Run() {
	//Vet(box)
	//input := box.Input
	//output := box.Output
	//str := box.stream

	//var count [3]int
	//prefComp := X
	//prefChan := Input[prefComp]

	//for {
	//	select{
	//		case dev := <-prefChan: sendToHost(dev)
	//	}	
	//}
}
