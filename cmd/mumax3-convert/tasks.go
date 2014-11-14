package main

import (
	"runtime"
	"sync"
)

var tasks chan func()
var taskWG sync.WaitGroup

const TaskCap = 100

func Queue(f func()) {
	if tasks == nil {
		tasks = make(chan func(), TaskCap)
		startWorkers()
	}

	taskWG.Add(1)
	tasks <- func() { defer taskWG.Add(-1); f() }
}

func Wait() {
	taskWG.Wait()
}

func startWorkers() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	nCPU := runtime.GOMAXPROCS(-1)
	for i := 0; i < nCPU+1; i++ {
		go func() {
			for f := range tasks {
				f()
			}
		}()
	}
}
