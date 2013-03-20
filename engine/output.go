package engine

// asynchronous output server.

import (
	"code.google.com/p/mx3/data"
	"log"
)

// returns a zeroed GPU buffer to "add" a quant to.
// then it should be added to the sum.
// then this buffer should be gosaved.
func OutputBuffer(nComp int) *data.Slice {
	return nil
}

// asynchronously copy slice to host to save it.
// return gpu buffer to pool as soon as copied.
func GoSaveAndRecycle(s *data.Slice, fname string) {
	log.Println("save", fname)
}
