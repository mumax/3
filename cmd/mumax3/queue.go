package main

import (
	"container/list"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"log"
	"os"
	"os/exec"
)

var queued, finished Queue

type Queue struct {
	list *list.List
}

func (q *Queue) Push(j *Job) {
	if q.list == nil {
		q.list = list.New()
	}
	q.list.PushBack(j)
}

func (q *Queue) Pop() *Job {
	f := q.list.Front()
	if f == nil {
		return nil
	} else {
		return q.list.Remove(f).(*Job)
	}
}

type Job struct {
	inFile string
}

func NewJob(inFile string) *Job {
	return &Job{inFile: inFile}
}

type Provider struct {
	gpu int
	job *Job
}

func (p *Provider) Run(j *Job, idle chan *Provider) {
	p.job = j

	gpu := fmt.Sprint("-gpu=", p.gpu)
	cmd := exec.Command(os.Args[0], gpu, j.inFile)
	Log("exec", os.Args[0], gpu, j.inFile)
	err := cmd.Run()
	if err != nil {
		Log(j, err)
	}

	p.job = nil
	idle <- p
}

var (
	providers []*Provider
	idle      chan *Provider
)

func initProviders() {
	nGpu := cu.DeviceGetCount()
	providers = make([]*Provider, nGpu)
	idle = make(chan *Provider, nGpu)
	for i := range providers {
		providers[i] = &Provider{gpu: i}
		idle <- providers[i]
	}
}

func runQueue(files []string) {
	initProviders()
	if len(providers) == 0 {
		log.Fatal("no GPUs available")
	}

	for _, f := range files {
		queued.Push(NewJob(f))
	}

	for j := queued.Pop(); j != nil; j = queued.Pop() {
		p := <-idle
		go p.Run(j, idle)
	}
}

func Log(msg ...interface{}) {
	log.Println(msg...)
}
