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
	Input  [3]<-chan GpuFloats
	Output [3][]chan<- []float32
	stream cu.Stream
	score  [3]int
	comp1  int // preferred component to send
	comp2  int // next preferred component to send
}

func NewFromGpu3Par() *FromGpu3Par {
	box := new(FromGpu3Par)
	box.stream = cu.StreamCreate()
	box.comp1 = X
	box.comp2 = Y
	Register(box)
	return box
}

func (box *FromGpu3Par) Run() {
	Vet(box)
	input := box.Input

	for {
		chan1 := input[box.comp1]
		chan2 := input[box.comp2]

		select {
		case dev := <-chan1:
			box.sendToHost(dev, box.comp1)
		default:
			select {
			case dev := <-chan2:
				box.sendToHost(dev, box.comp2)
			default:
				select {
				case x := <-input[X]:
					box.sendToHost(x, X)
				case y := <-input[X]:
					box.sendToHost(y, Y)
				case z := <-input[X]:
					box.sendToHost(z, Z)
				}
			}
		}
	}
}

func (box *FromGpu3Par) sendToHost(in GpuFloats, comp int) {
	sendToHost(in, box.Output[X], box.stream)
	box.score[comp]++

	box.comp1 = (comp + 1) % 3
	box.comp2 = (comp + 2) % 3

	Debug("score:", box.score, "comp1=", box.comp1, ",comp2=", box.comp2)
}
