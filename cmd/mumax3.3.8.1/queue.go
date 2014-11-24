package main

// File que for distributing multiple input files over GPUs.

import (
	"fmt"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/engine"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

func RunQueue(files []string) {
	s := NewStateTab(files)
	s.PrintTo(os.Stdout)
	go s.ListenAndServe(*flag_port)
	s.Run()
	os.Exit(int(exitStatus))
}

// StateTab holds the queue state (list of jobs + statuses).
// All operations are atomic.
type stateTab struct {
	lock sync.Mutex
	jobs []job
	next int
}

// Job info.
type job struct {
	inFile  string // input file to run
	webAddr string // http address for gui of running process
	uid     int
}

// NewStateTab constructs a queue for the given input files.
// After construction, it is accessed atomically.
func NewStateTab(inFiles []string) *stateTab {
	s := new(stateTab)
	s.jobs = make([]job, len(inFiles))
	for i, f := range inFiles {
		s.jobs[i] = job{inFile: f, uid: i}
	}
	return s
}

// StartNext advances the next job and marks it running, setting its webAddr to indicate the GUI url.
// A copy of the job info is returned, the original remains unmodified.
// ok is false if there is no next job.
func (s *stateTab) StartNext(webAddr string) (next job, ok bool) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if s.next >= len(s.jobs) {
		return job{}, false
	}
	s.jobs[s.next].webAddr = webAddr
	jobCopy := s.jobs[s.next]
	s.next++
	return jobCopy, true
}

// Finish marks the job with j's uid as finished.
func (s *stateTab) Finish(j job) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.jobs[j.uid].webAddr = ""
}

// Runs all the jobs in stateTab.
func (s *stateTab) Run() {
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

type atom int32

func (a *atom) set(v int) { atomic.StoreInt32((*int32)(a), int32(v)) }
func (a *atom) get() int  { return int(atomic.LoadInt32((*int32)(a))) }

var exitStatus atom = 0

func run(inFile string, gpu int, webAddr string) {
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	cmd := exec.Command(os.Args[0], cacheFlag, gpuFlag, httpFlag, inFile)
	log.Println(os.Args[0], cacheFlag, gpuFlag, httpFlag, inFile)
	err := cmd.Run()
	if err != nil {
		log.Println(inFile, err)
		exitStatus.set(1)
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

func (s *stateTab) PrintTo(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	for i, j := range s.jobs {
		fmt.Fprintf(w, "%3d %v %v\n", i, j.inFile, j.webAddr)
	}
}

func (s *stateTab) RenderHTML(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	fmt.Fprintln(w, ` 
<!DOCTYPE html> <html> <head> 
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	<meta http-equiv="refresh" content="1">
`+engine.CSS+`
	</head><body>
	<span style="color:gray; font-weight:bold; font-size:1.5em"> mumax<sup>3</sup> queue status </span><br/>
	<hr/>
	<pre>
`)

	hostname := "localhost"
	hostname, _ = os.Hostname()
	for _, j := range s.jobs {
		if j.webAddr != "" {
			fmt.Fprint(w, `<b>`, j.uid, ` <a href="`, "http://", hostname+j.webAddr, `">`, j.inFile, " ", j.webAddr, "</a></b>\n")
		} else {
			fmt.Fprint(w, j.uid, " ", j.inFile, "\n")
		}
	}

	fmt.Fprintln(w, `</pre><hr/></body></html>`)
}

func (s *stateTab) ListenAndServe(addr string) {
	http.Handle("/", s)
	go http.ListenAndServe(addr, nil)
}

func (s *stateTab) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.RenderHTML(w)
}
