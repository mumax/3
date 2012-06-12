package mm

import (
	//"io"
	"bufio"
	"fmt"
	"log"
	. "nimble-cube/nc"
	"os"
)

//CONCEPT: Send interfaces over the fannels
//Slice, Slice3, Scalar, Vector, GPUSlice, ...
//Use type info for nice output, recycling decissions...

type TableBox struct {
	Input  <-chan float64
	Time   <-chan float64 "time"
	writer *bufio.Writer
}

func NewTableBox(file string) *TableBox {
	box := new(TableBox)
	out, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	CheckIO(err)
	box.writer = bufio.NewWriter(out)
	Register(box)
	return box
}

func (box *TableBox) Run() {
	defer box.Close()
	for {
		time := RecvFloat64(box.Time)
		value := RecvFloat64(box.Input)
		fmt.Fprintln(box.writer, time, "\t", value)
	}
}

func (box *TableBox) Close() {
	log.Println("closing", box.writer)
	box.writer.Flush()
}
