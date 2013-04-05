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
func goSave(fname string, output *data.Slice, t float64, unlockOutput func()) {
	if dlQue == nil {
		dlQue = make(chan dlTask)
		saveQue = make(chan saveTask)
		hBuf = make(chan *data.Slice, maxOutputQueLen)
		go runDownloader()
		go runSaver()
	}
	dlQue <- dlTask{fname, output, t, unlockOutput}
}

var (
	dlQue   chan dlTask       // passes download requests from goSave to runDownloader
	saveQue chan saveTask     // passes save requests from runDownloader to runSaver
	hBuf    chan *data.Slice  // pool of page-locked host buffers for save queue
	done    = make(chan bool) // marks output server is completely done after closing dlQue
)

// download task
type dlTask struct {
	fname        string
	output       *data.Slice
	time         float64
	unlockOutput func()
}

// save task
type saveTask dlTask

// At most this many outputs can be queued for asynchronous saving to disk.
const maxOutputQueLen = 16

var nOutBuf int // number of output buffers actually in use (<= maxOutputQueLen)

// returns host buffer for storing output before being flushed to disk.
// takes one from the pool or allocates a new one when the pool is empty
// and less than maxOutputQueLen buffers already are in use.
func hostbuf() *data.Slice {
	select {
	case b := <-hBuf:
		return b
	default:
		if nOutBuf < maxOutputQueLen {
			nOutBuf++
			return cuda.NewUnifiedSlice(3, mesh)
		}
	}
	panic("unreachable")
}

// continuously takes download tasks and queues corresponding save tasks.
// the downloader queue is not buffered and we want to use at most one GPU
// output buffer. Only one PCIe download at a time can proceed anyway.
func runDownloader() {
	cuda.LockThread()

	for t := range dlQue {
		h := hostbuf()
		data.Copy(h, t.output) // output is already locked
		t.unlockOutput()
		saveQue <- saveTask{t.fname, h, t.time, func() { cuda.Memset(h, 0, 0, 0); hBuf <- h }}
	}
	close(saveQue)
}

// continuously takes save tasks and flushes them to disk.
// the save queue can accommodate many outputs (stored on host).
// the rather big queue allows big output bursts to be concurrent.
func runSaver() {
	for t := range saveQue {
		data.MustWriteFile(t.fname, t.output, t.time)
		t.unlockOutput() // typically puts buffer back in pool
	}
	done <- true
}

// finalizer function called upon program exit.
// waits until all asynchronous output has been saved.
func drainOutput() {
	if dlQue != nil {
		log.Println("flushing output")
		close(dlQue)
		<-done
	}
}
