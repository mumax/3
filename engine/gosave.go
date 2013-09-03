package engine

import (
	"code.google.com/p/mx3/data"
	"log"
)

var (
	saveQue chan saveTask     // passes save requests from runDownloader to runSaver
	done    = make(chan bool) // marks output server is completely done after closing dlQue
	nOutBuf int               // number of output buffers actually in use (<= maxOutputQueLen)
)

const maxOutputQueLen = 16 // number of outputs that can be queued for asynchronous I/O.

func AsyncSave(fname string, s *data.Slice, time float64) {
	initQue()
	saveQue <- saveTask{fname, s, time}
}

func initQue() {
	if saveQue == nil {
		saveQue = make(chan saveTask)
		go runSaver()
	}
}

// output task
type saveTask struct {
	fname  string
	output *data.Slice // needs to be recylced
	time   float64
}

// continuously takes save tasks and flushes them to disk.
// the save queue can accommodate many outputs (stored on host).
// the rather big queue allows big output bursts to be concurrent.
func runSaver() {
	for t := range saveQue {
		data.MustWriteFile(t.fname, t.output, t.time)
		//hBuf <- t.output
	}
	done <- true
}

// finalizer function called upon program exit.
// waits until all asynchronous output has been saved.
func drainOutput() {
	if saveQue != nil {
		log.Println("flushing output")
		close(saveQue)
		<-done
	}
}
