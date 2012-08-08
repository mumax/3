package nc

import (
	"github.com/barnex/cuda4/cu"
)

// Copies a 3-vector from GPU,
// interleaving the components as much as possible. 
// I.e.: Tries to copy X,Y,Z X,Y,Z X,Y,Z, ...
// However, if a component is ready to be sent it is skipped,
// so the box will not idle.

//  Source should not be eager, push at most 1 frame at a time -> test files
//  ISSUE: SHOULD BE AWARE OF BUFFER, NOT OVERFULL BUFFER.?

type FromGpu3Par struct {
	Input  [3]<-chan GpuBlock
	Output [3][]chan<- Block
	//	score  [3]int
	//	comp1  int // preferred component to send
	//	comp2  int // next preferred component to send
}

func NewFromGpu3Par() *FromGpu3Par {
	box := new(FromGpu3Par)
	Register(box)
	return box
}

func (box *FromGpu3Par) Run() {
	LockCudaThread()
	Vet(box)
	input := box.Input
	str := cu.StreamCreate()
	defer str.Destroy()

	for {
		x := RecvGpu(input[X])
		Debug("send X")
		sendToHost(x, box.Output[X], str)
		y := RecvGpu(input[Y])
		Debug("send Y")
		sendToHost(y, box.Output[Y], str)
		z := RecvGpu(input[Z])
		Debug("send Z")
		sendToHost(z, box.Output[Z], str)
	}

}

//func (box *FromGpu3Par) Run() {
//	Vet(box)
//	input := box.Input
//
//	for {
//		chan1 := input[box.comp1]
//		chan2 := input[box.comp2]
//
//		select {
//		case dev := <-chan1:
//			box.sendToHost(dev, box.comp1)
//		default:
//			select {
//			case dev := <-chan2:
//				box.sendToHost(dev, box.comp2)
//			default:
//				select {
//				case x := <-input[X]:
//					box.sendToHost(x, X)
//				case y := <-input[X]:
//					box.sendToHost(y, Y)
//				case z := <-input[X]:
//					box.sendToHost(z, Z)
//				}
//			}
//		}
//	}
//}

//func (box *FromGpu3Par) sendToHost(in GpuFloats, comp int) {
//	sendToHost(in, box.Output[X], box.stream)
//	box.score[comp]++
//
//	box.comp1 = (comp + 1) % 3
//	box.comp2 = (comp + 2) % 3
//
//	Debug("score:", box.score, "comp1=", box.comp1, ",comp2=", box.comp2)
//}
