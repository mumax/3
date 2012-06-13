package mm

import (
	"bufio"
	"fmt"
	"log"
	. "nimble-cube/nc"
	"os"
)

// Writes time + data table to a file.
type TableBox struct {
	Input  <-chan float64
	Time   <-chan float64 "time"
	writer *bufio.Writer
}

func NewTableBox(file string) *TableBox {
	box := new(TableBox)
	out, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	PanicErr(err)
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
