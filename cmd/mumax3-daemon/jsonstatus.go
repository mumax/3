package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
)

type Status struct {
	Name      string
	Status    int // -1: failed, 0: finished, 1:running
	StartTime time.Time
	Runtime   time.Duration
	Node      string
	Gpu       int
}

func saveStartStatus(job *Job, lockdir string) {
	var status Status
	status.Name = fmt.Sprint(job.Command, job.Args)
	status.Node = *flag_host
	status.Gpu = *flag_gpu
	status.StartTime = time.Now()
	status.Status = 1

	out, err := os.OpenFile(lockdir+"/status.json", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		log.Println(err)
		return
	}
	defer out.Close()

	json.NewEncoder(out).Encode(&status)

}

func saveStopStatus(job *Job, lockdir string, exitstat int) {

	in, err := os.Open(lockdir + "/status.json")
	if err != nil {
		log.Println(err)
		return
	}

	var status Status
	err2 := json.NewDecoder(in).Decode(&status)
	in.Close()
	if err2 != nil {
		log.Println(err2)
		return
	}

	status.Runtime = time.Since(status.StartTime)
	status.Status = exitstat

	out, err := os.OpenFile(lockdir+"/status.json", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		log.Println(err)
	}
	defer out.Close()

	json.NewEncoder(out).Encode(&status)
}
