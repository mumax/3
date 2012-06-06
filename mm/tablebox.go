package mm

import (
//"io"
//. "nimble-cube/nc"
//"os"
)

// Outputs data to table
// reduces slice data on the fly

//CONCEPT: Send interfaces over the fannels
//Slice, Slice3, Scalar, Vector, GPUSlice, ...
//Use type info for nice output, recycling decissions...

//type TableBox3 struct {
//	input  FanOutScalar
//	time   FanOutScalar
//	writer io.Writer
//}
//
//func NewTableBox(file string) *TableBox {
//	box := new(TableBox)
//	var err error
//	box.writer, err = os.Open(file)
//	CheckIO(err)
//	return box
//}
//
//func (box *TableBox) Run() {
//	ok := true
//	for ok{
//		var time float32
//		time, ok = <-box.time
//
//		
//
//		fmt.Fprintln(box.writer, time, "\t", 
//	}
//}
