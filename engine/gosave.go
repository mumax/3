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

// Asynchronously save output to file. Calls unlockOutput() to unlock output read lock.
func GoSave(fname string, output *data.Slice, t float64, unlockOutput func()) {
	if outputrequests == nil {
		outputrequests = make(chan outTask)
		go RunOutputServer()
	}
	outputrequests <- outTask{fname, output, t, unlockOutput}
}

var (
	outputrequests chan outTask      // pipes output requests from GoSave to RunOutputServer
	done           = make(chan bool) // marks output server is completely done after closing outputrequests
)

// output task
type outTask struct {
	fname        string
	output       *data.Slice
	time         float64
	unlockOutput func()
}

// output goroutine to be run concurrently with simulation.
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

// finalizer function called upon program exit.
// waits until all asynchronous output has been saved.
func drainOutput() {
	log.Println("flushing output")
	close(outputrequests)
	<-done
}
