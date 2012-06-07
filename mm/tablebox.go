package mm

import (
	"fmt"
	"io"
	. "nimble-cube/nc"
	"os"
)

//CONCEPT: Send interfaces over the fannels
//Slice, Slice3, Scalar, Vector, GPUSlice, ...
//Use type info for nice output, recycling decissions...

type TableBox struct {
	input  []<-chan float64
	time   <-chan float64
	writer io.WriteCloser
}

func NewTableBox(file string) *TableBox {
	box := new(TableBox)
	var err error
	box.writer, err = os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	CheckIO(err)
	return box
}

func (box *TableBox) Run() {
	for {
		var time float64
		time, ok := <-box.time
		if !ok {
			box.writer.Close()
			// sync
			break
		}
		fmt.Fprint(box.writer, time)
		for _, in := range box.input {
			value := <-in
			fmt.Fprint(box.writer, "\t", value)
		}
		fmt.Fprintln(box.writer)
	}
}
