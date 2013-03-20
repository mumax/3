package engine

// asynchronous output server.

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
)

const nbuf = 1 // todo: use a few more if we have enough memory

var (
	gpubuf  chan *data.Slice
	hostbuf chan *data.Slice
)

// returns a zeroed GPU buffer to "add" a quant to.
// then it should be added to the sum.
// then this buffer should be gosaved.
func outputBuffer(nComp int) *data.Slice {
	initOutBuf()
	buf := <-gpubuf
	cuda.Memset(buf, 0, 0, 0)
	return buf
}

// asynchronously copy slice to host to save it.
// return gpu buffer to pool as soon as copied.
func saveAndRecycle(s *data.Slice, fname string, time float64) {
	host := <-hostbuf
	data.Copy(host, s) // async
	gpubuf <- s
	data.MustWriteFile(fname, host, time)
	hostbuf <- host
}

func initOutBuf() {
	if gpubuf == nil {
		gpubuf = make(chan *data.Slice, nbuf)
		hostbuf = make(chan *data.Slice, nbuf)
		for i := 0; i < nbuf; i++ {
			gpubuf <- cuda.NewSlice(3, mesh)
			hostbuf <- cuda.NewUnifiedSlice(3, mesh)
		}
	}
}

func drainOutput() {
	if gpubuf != nil {
		log.Println("flushing output")
		for i := 0; i < nbuf; i++ {
			<-gpubuf
			<-hostbuf
		}
	}
}
