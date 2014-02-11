package engine

// Asynchronous I/O queue flushes data to disk while simulation keeps running.
// See save.go, autosave.go

var (
	SaveQue chan func()       // passes save requests from runDownloader to runSaver
	done    = make(chan bool) // marks output server is completely done after closing dlQue
	nOutBuf int               // number of output buffers actually in use (<= maxOutputQueLen)
)

const maxOutputQueLen = 16 // number of outputs that can be queued for asynchronous I/O.

func initQue() {
	if SaveQue == nil {
		SaveQue = make(chan func())
		go runSaver()
	}
}

// Continuously executes tasks the from SaveQue channel.
func runSaver() {
	for f := range SaveQue {
		f()
	}
	done <- true
}

// Finalizer function called upon program exit.
// Waits until all asynchronous output has been saved.
func drainOutput() {
	if SaveQue != nil {
		close(SaveQue)
		<-done
	}
}
