package main

const N_WORKER = 2

var workers = make(chan func())

func StartWorkers() {
	for i := 0; i < N_WORKER; i++ {
		go RunWorker()
	}
}

func RunWorker() {
	for {
		(<-workers)()
	}
}
