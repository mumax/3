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
	Input  <-chan float64
	Time   <-chan float64
	writer io.WriteCloser
}

func NewTableBox(file string, quant string) *TableBox {
	box := new(TableBox)
	var err error
	box.writer, err = os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	CheckIO(err)

	ConnectToQuant(box, &box.Input, quant)

	return box
}

func (box *TableBox) Run() {
	defer box.writer.Close()
	for {
		time := RecvFloat64(box.Time)
		fmt.Fprint(box.writer, time)
		value := RecvFloat64(box.Input)
		fmt.Fprint(box.writer, "\t", value)
		fmt.Fprintln(box.writer)
	}
}
