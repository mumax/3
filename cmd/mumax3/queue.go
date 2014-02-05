package main

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
)

func RunQueue(files []string) {
	s := NewStateTab(files)
	s.PrintTo(os.Stdout)
	runStateTab(s)
}

// StateTab holds the queue state (list of jobs + statuses).
// All operations are atomic.
type StateTab struct {
	lock sync.Mutex
	jobs []Job
	next int
}

// Job holds a copy of job info.
// Modifying the fields has no effect.
type Job struct {
	inFile  string // input file to run
	webAddr string // http address for gui of running process
	uid     int
}

// NewStateTab constructs a queue for the given input files.
// After construction, it is accessed atomically.
func NewStateTab(inFiles []string) *StateTab {
	s := new(StateTab)
	s.jobs = make([]Job, len(inFiles))
	for i, f := range inFiles {
		s.jobs[i] = Job{inFile: f, uid: i}
	}
	return s
}

// StartNext advances the next job and marks it running, setting its webAddr to indicate the GUI url.
// A copy of the job info is returned, the original remains unmodified.
// ok is false if there is no next job.
func (s *StateTab) StartNext(webAddr string) (next Job, ok bool) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if s.next >= len(s.jobs) {
		return Job{}, false
	}
	s.jobs[s.next].webAddr = webAddr
	jobCopy := s.jobs[s.next]
	s.next++
	return jobCopy, true
}

// Finish marks the job with j's uid as finished.
func (s *StateTab) Finish(j Job) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.jobs[j.uid].webAddr = ""
}

func (s *StateTab) PrintTo(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	for i, j := range s.jobs {
		fmt.Fprintf(w, "%3d %v %v\n", i, j.inFile, j.webAddr)
	}
}

func runStateTab(s *StateTab) {
	nGPU := cu.DeviceGetCount()
	idle := initGPUs(nGPU)
	for {
		gpu := <-idle
		addr := fmt.Sprint(":", 35368+gpu)
		j, ok := s.StartNext(addr)
		if !ok {
			break
		}
		go func() {
			run(j.inFile, gpu, j.webAddr)
			s.Finish(j)
			idle <- gpu
		}()
	}
	// drain remaining tasks (one already done)
	for i := 1; i < nGPU; i++ {
		<-idle
	}
}

func run(inFile string, gpu int, webAddr string) {
	gpuFlag := fmt.Sprint("-gpu=", gpu)
	cmd := exec.Command(os.Args[0], gpuFlag, inFile)
	log.Println("exec", os.Args[0], gpuFlag, inFile)
	err := cmd.Run()
	if err != nil {
		log.Println(inFile, err)
	}
}

func initGPUs(nGpu int) chan int {
	if nGpu == 0 {
		log.Fatal("no GPUs available")
		panic(0)
	}
	idle := make(chan int, nGpu)
	for i := 0; i < nGpu; i++ {
		idle <- i
	}
	return idle
}

//func(o*Overlord)PushJob(f string){
//	o.queued.Push(NewJob(f))
//}
//
//type Queue struct {
//	list *list.List
//}
//
//func (q *Queue) Push(j Job) {
//	q.l****
//	if q.list == nil {
//		q.list = list.New()
//	}
//	q.list.PushBack(j)
//}
//
//func (q *Queue) Pop() Job {
//	f := q.list.Front()
//	if f == nil {
//		return nil
//	} else {
//		return q.list.Remove(f).(Job)
//	}
//}
//
//type Job struct {
//	inFile string
//}
//
//func NewJob(inFile string) *Job {
//	return &Job{inFile: inFile}
//}
//
//type Provider struct {
//	gpu int
//}
//
//
//
//
//func runQueue(files []string) {
//	o := Overlord{rpc: make(chan func())}
//	o.initProviders()
//	o.GoServeHTTP()
//	for _, f := range files {
//		o.PushJob(f)
//	}
//	o.RunQueue()
//}
//
//func(o*Overlord)RunQueue(){
//	for j := o.queued.Pop(); j != nil; j = o.queued.Pop() {
//		select{
//		case p := <-o.idle:
//		go func(j*Job){
//			p.Run(j)
//			o.finished.Push(j)
//		}(j)
//		case f := <- o.rpc:
//			f()
//		}
//	}
//}
//
//func Log(msg ...interface{}) {
//	log.Println(msg...)
//}
