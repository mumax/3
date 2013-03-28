package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
)

var _hostbuf *data.Synced

// returns host buffer for output. TODO: have more than one.
func hostbuf() *data.Synced {
	if _hostbuf == nil {
		log.Println("allocating host output buffer")
		_hostbuf = data.NewSynced(cuda.NewUnifiedSlice(3, mesh))
	}
	return _hostbuf
}

func GoSave(fname string, output *data.Slice, t float64, unlockOutput func()) {
	if outputrequests == nil {
		outputrequests = make(chan outTask)
		go RunOutputServer()
	}
	outputrequests <- outTask{fname, output, t, unlockOutput}
}

var (
	outputrequests chan outTask
	done           = make(chan bool)
)

type outTask struct {
	fname        string
	output       *data.Slice
	time         float64
	unlockOutput func()
}

func RunOutputServer() {
	cuda.LockThread()

	for t := range outputrequests {
		H := hostbuf()
		h := H.Write()

		data.Copy(h, t.output)

		t.unlockOutput()

		data.MustWriteFile(t.fname, h, t.time)

		H.WriteDone()
	}
	done <- true
}

func drainOutput() {
	log.Println("flushing output")
	close(outputrequests)
	<-done
}
