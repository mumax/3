package nc

import (
	"github.com/barnex/cuda4/cu"
)

// Copies a 3-vector to GPU,
// component by-component as much possible.
// I.e.: Tries to copy first all X, then Y and finally Z components.
// However, if no X component is ready to be sent, the box will not idle
// but rather start copying the next component.
type ToGpu3SeqBox struct {
	Input  [3]<-chan Block
	Output [3][]chan<- GpuBlock
	stream cu.Stream
}

func NewToGpu3SeqBox() *ToGpu3SeqBox {
	box := new(ToGpu3SeqBox)
	box.stream = cu.StreamCreate()
	Register(box)
	return box
}

func (box *ToGpu3SeqBox) Run() {
	Vet(box)
	input := box.Input
	output := box.Output
	str := box.stream
	for {
		//selecting over input channels is safe according to postman.
		select {
		case x := <-input[X]:
			Debug("sending X")
			sendToGpu(x, output[X], str)
		default:
			select {
			case y := <-input[Y]:
				Debug("sending Y")
				sendToGpu(y, output[Y], str)
			default:
				select {
				case x := <-input[X]:
					Debug("sending X")
					sendToGpu(x, output[X], str)
				case y := <-input[Y]:
					Debug("sending Y")
					sendToGpu(y, output[Y], str)
				case z := <-input[Z]:
					Debug("sending Z")
					sendToGpu(z, output[Z], str)
				}
			}
		}
	}
}
