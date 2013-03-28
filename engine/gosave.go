package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
)

// Asynchronously save output to file. Calls unlockOutput() to unlock output read lock.
// This function returns as soon as downloading output to host can start (usually immediately).
// During the download, further computations can continue. When the download is done, unlockOutput()
// is called so that the output buffer can be written again. Then the output host copy is
// queued for saving to disk. 
func GoSave(fname string, output *data.Slice, t float64, unlockOutput func()) {
	if dlQue == nil {
		dlQue = make(chan dlTask)
		saveQue = make(chan saveTask)
		hBuf = make(chan *data.Slice, MaxOutputQueLen)
		go RunDownloader()
		go RunSaver()
	}
	dlQue <- dlTask{fname, output, t, unlockOutput}
}

var (
	dlQue   chan dlTask       // pipes download requests from GoSave to RunDownloader
	saveQue chan saveTask     // pipes save requests from RunDownloader to RunSaver
	hBuf    chan *data.Slice  // pool of page-locked host buffers
	done    = make(chan bool) // marks output server is completely done after closing dlQue
)

// download task
type dlTask struct {
	fname        string
	output       *data.Slice
	time         float64
	unlockOutput func()
}

type saveTask dlTask

// At most this many outputs can be queued for asynchronous saving to disk.
const MaxOutputQueLen = 16

var nOutBuf int // number of output buffers actually in use (<= MaxOutputQueLen)

// returns host buffer for output. TODO: have more than one.
func hostbuf() *data.Slice {
	select {
	case b := <-hBuf:
		return b
	default:
		if nOutBuf < MaxOutputQueLen {
			nOutBuf++
			//util.DashExit()
			//log.Println("using", nOutBuf, "host output buffers")
			return cuda.NewUnifiedSlice(3, mesh)
		}
	}
	panic("unreachable")
	return nil
}

// output goroutine to be run concurrently with simulation.
func RunDownloader() {
	cuda.LockThread()

	for t := range dlQue {
		h := hostbuf()
		data.Copy(h, t.output)
		t.unlockOutput()
		saveQue <- saveTask{t.fname, h, t.time, func() { hBuf <- h }}
	}
	close(saveQue)
}

func RunSaver() {
	for t := range saveQue {
		data.MustWriteFile(t.fname, t.output, t.time)
		t.unlockOutput()
	}
	done <- true
}

// finalizer function called upon program exit.
// waits until all asynchronous output has been saved.
func drainOutput() {
	log.Println("flushing output")
	close(dlQue)
	<-done
}
