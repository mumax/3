package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/timer"
)

// Asynchronous I/O queue flushes data to disk while simulation keeps running.
// See save.go, autosave.go

var (
	saveQue chan func()       // passes save requests from runDownloader to runSaver
	done    = make(chan bool) // marks output server is completely done after closing dlQue
	nOutBuf int               // number of output buffers actually in use (<= maxOutputQueLen)
)

const maxOutputQueLen = 16 // number of outputs that can be queued for asynchronous I/O.

func init() {
	saveQue = make(chan func())
	go runSaver()
}

func queOutput(f func()) {
	if cuda.Synchronous {
		timer.Start("io")
	}
	saveQue <- f
	if cuda.Synchronous {
		timer.Stop("io")
	}
}

// Continuously executes tasks the from SaveQue channel.
func runSaver() {
	for f := range saveQue {
		f()
	}
	done <- true
}

// Finalizer function called upon program exit.
// Waits until all asynchronous output has been saved.
func drainOutput() {
	if saveQue != nil {
		if cuda.Synchronous {
			timer.Start("io")
		}
		close(saveQue)
		<-done
		if cuda.Synchronous {
			timer.Stop("io")
		}
	}
}
