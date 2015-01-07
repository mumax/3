package main

/*
	Compute service runs jobs on this node's GPUs, if any.
*/

import (
	"fmt"
	"io"
	"log"
	"os/exec"
	"strings"
	"time"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

var (
	MumaxVersion string
	GPUs         []string
	Processes    = make(map[string]*Process) // job id -> process
)

// Process is a running simulation process
type Process struct {
	*exec.Cmd
	Start     time.Time
	Out       io.WriteCloser
	ID        string
	OutputURL string
	GUI       string
	Killed    bool
}

func (p *Process) Host() string {
	return JobHost(p.OutputURL)
}

// Runs a compute service on this node, if GPUs are available.
// The compute service asks storage nodes for a job, runs it,
// saves results over httpfs and notifies storage when ready.
func RunComputeService() {

	if len(GPUs) == 0 {
		return
	}

	// queue of available GPU numbers
	idle := make(chan int, len(GPUs))
	for i := range GPUs {
		idle <- i
	}

	for {
		gpu := <-idle // take an available GPU
		GUIAddr := fmt.Sprint(thisHost+":", GUI_PORT+gpu)
		ID := WaitForJob() // take an available job
		go func() {

			defer func() {
				// remove from "running" list
				WLock()
				delete(Processes, ID)
				WUnlock()
				// add GPU number back to idle stack
				idle <- gpu
			}()

			p := NewProcess(ID, gpu, GUIAddr)
			if p == nil {
				return
			}

			WLock()
			Processes[ID] = p
			WUnlock()

			p.Run()

			_, err := RPCCall(JobHost(ID), "UpdateJob", ID)
			if err != nil {
				log.Println(err)
			}

		}()
	}
}

func WaitForJob() string {
	ID := FindJob()
	for ID == "" {
		time.Sleep(2 * time.Second) // TODO: don't poll
		ID = FindJob()
	}
	return ID
}

func FindJob() string {

	// quickly list peers first
	RLock()
	p := make([]string, 0, len(peers))
	for addr, _ := range peers {
		p = append(p, addr)
	}
	RUnlock()
	// TODO: pick peers fairly

	// then do slow RPC calls without blocking the rest of the program
	for _, addr := range p {
		ID, _ := RPCCall(addr, "GiveJob", thisAddr)
		if ID != "" {
			return ID
		}
	}
	return ""
}

// RPC-callable function kills job corresponding to given job id.
// The job has to be running on this node.
func Kill(id string) string {
	log.Println("KILL", id)

	WLock() // modifies Cmd state
	defer WUnlock()

	job := Processes[id]
	if job == nil {
		return fmt.Sprintf("kill %v: job not running.", id)
	}
	job.Killed = true
	err := job.Cmd.Process.Kill()
	if err != nil {
		return err.Error()
	}
	return "" // OK
}

// prepare exec.Cmd to run mumax3 compute process
func NewProcess(ID string, gpu int, webAddr string) *Process {
	// prepare command
	inputURL := "http://" + ID
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	forceFlag := `-f=0`
	cmd := exec.Command(command, gpuFlag, httpFlag, cacheFlag, forceFlag, inputURL)

	// Pipe stdout, stderr to log file over httpfs
	outDir := util.NoExt(inputURL) + ".out"
	errMkdir := httpfs.Mkdir(outDir)
	if errMkdir != nil {
		SetJobError(ID, errMkdir)
		log.Println("makeProcess", errMkdir)
		j := JobByName(ID)
		if j != nil {
			j.Reque()
		}
		return nil
	}

	out, errD := httpfs.Create(outDir + "/stdout.txt")
	if errD != nil {
		SetJobError(ID, errD)
		log.Println("makeProcess", errD)
		j := JobByName(ID)
		if j != nil {
			j.Reque()
		}
		return nil
	}
	cmd.Stderr = out
	cmd.Stdout = out

	return &Process{ID: ID, Cmd: cmd, Start: time.Now(), Out: out, OutputURL: OutputDir(inputURL), GUI: webAddr}
}

func (p *Process) Run() {

	log.Println("=> exec  ", p.Path, p.Args)

	defer p.Out.Close()

	httpfs.Put(p.OutputURL+"host", []byte(thisAddr))

	startTime := AskTime(p.Host())
	httpfs.Put(p.OutputURL+"start", []byte(startTime.Format(time.UnixDate)))

	WLock()               // Cmd.Start() modifies state
	err1 := p.Cmd.Start() // err?
	WUnlock()
	if err1 != nil {
		SetJobError(p.ID, err1)
	}

	timeOffset := time.Now().Sub(startTime) // our clock is most likely out-of-sync with host
	tick := time.NewTicker(KeepaliveInterval)

	// need initial alive in case watchdog sniffs between start and first alive tick
	httpfs.Put(p.OutputURL+"alive", []byte(time.Now().Add(timeOffset).Format(time.UnixDate)))
	go func() {
		for t := range tick.C {
			httpfs.Put(p.OutputURL+"alive", []byte(t.Add(timeOffset).Format(time.UnixDate)))
		}
	}()

	err2 := p.Cmd.Wait()
	if err1 == nil && err2 != nil {
		SetJobError(p.ID, err2)
	}
	tick.Stop()

	status := -1

	// TODO: determine proper status number
	if err1 != nil || err2 != nil {
		log.Println(p.Path, p.Args, err1, err2)
		status = 1
	} else {
		status = 0
	}

	if p.Killed {
		httpfs.Put(p.OutputURL+"killed", []byte(time.Now().Format(time.UnixDate)))
	} else {
		httpfs.Put(p.OutputURL+"exitstatus", []byte(fmt.Sprint(status)))
	}

	stopTime := AskTime(p.Host())
	nanos := stopTime.Sub(startTime).Nanoseconds()
	httpfs.Put(p.OutputURL+"duration", []byte(fmt.Sprint(nanos)))

	if status == 0 {
		ret, err := RPCCall(p.Host(), "AddFairShare", JobUser(p.ID)+"/"+fmt.Sprint(nanos/1e9))
		if err != nil || ret != "" {
			log.Println("***ERR: AddFairShare", JobUser(p.ID), ret, err)
		}
	}

	return
}

func (p *Process) Duration() time.Duration { return Since(time.Now(), p.Start) }

func DetectGPUs() {
	if GPUs != nil {
		panic("multiple DetectGPUs() calls")
	}

	for i := 0; i < MAXGPU; i++ {
		gpuflag := fmt.Sprint("-gpu=", i)
		out, err := exec.Command(*flag_mumax, "-test", gpuflag).Output()
		if err == nil {
			info := string(out)
			if strings.HasSuffix(info, "\n") {
				info = info[:len(info)-1]
			}
			log.Println("gpu", i, ":", info)
			GPUs = append(GPUs, info)
		}
	}
}

func DetectMumax() {
	out, err := exec.Command(*flag_mumax, "-test", "-v").CombinedOutput()
	info := string(out)
	if err == nil {
		split := strings.SplitN(info, "\n", 2)
		version := split[0]
		log.Println("have", version)
		MumaxVersion = version
	} else {
		MumaxVersion = fmt.Sprint(*flag_mumax, "-test", ": ", err, info)
	}
}

// RPC-callable function, answers by this node's time
func WhatsTheTime(string) string {
	return time.Now().Format(time.UnixDate)
}

func AskTime(host string) time.Time {
	str, _ := RPCCall(host, "WhatsTheTime", "")
	return parseTime(str)
}

func parseTime(str string) time.Time {
	t, _ := time.Parse(time.UnixDate, str)
	return t
}
