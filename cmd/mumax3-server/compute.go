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
	Processes    = make(map[string]*Process)
)

type Process struct {
	*exec.Cmd
	Start time.Time
	Out   io.WriteCloser
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
		GUIAddr := fmt.Sprint(":", GUI_PORT+gpu)
		ID := WaitForJob() // take an available job
		go func() {

			p := NewProcess("http://"+ID, gpu, GUIAddr)

			WLock()
			Processes[ID] = p
			WUnlock()

			_, _ = RPCCall(JobHost(ID), "UpdateJob", ID) // update so we see .out appear on start

			p.Run()

			// remove from "running" list
			WLock()
			delete(Processes, ID)
			WUnlock()

			_, err := RPCCall(JobHost(ID), "UpdateJob", ID)
			if err != nil {
				log.Println(err)
			}

			// add GPU number back to idle stack
			idle <- gpu
		}()
	}
}

func (p *Process) Duration() time.Duration { return Since(time.Now(), p.Start) }

func WaitForJob() string {
	ID := FindJob()
	for ID == "" {
		time.Sleep(2 * time.Second) // TODO: don't poll
		ID = FindJob()
	}
	//log.Println("found job", ID)
	return ID
}

func FindJob() string {
	for addr, _ := range peers {
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
	WLock() // modifies Cmd state
	defer WUnlock()

	log.Println("KILL", id)
	job := Processes[id]
	if job == nil {
		return fmt.Sprintf("kill %v: job not running.", id)
	}
	err := job.Cmd.Process.Kill()
	if err != nil {
		return err.Error()
	}
	return "" // OK
}

// prepare exec.Cmd to run mumax3 compute process
func NewProcess(inputURL string, gpu int, webAddr string) *Process {
	// prepare command
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	forceFlag := `-f=0`
	cmd := exec.Command(command, gpuFlag, httpFlag, cacheFlag, forceFlag, inputURL)

	// Pipe stdout, stderr to log file over httpfs
	outDir := util.NoExt(inputURL) + ".out"
	httpfs.Mkdir(outDir)
	out, errD := httpfs.Create(outDir + "/stdout.txt")
	if errD != nil {
		log.Println("makeProcess", errD)
	}
	cmd.Stderr = out
	cmd.Stdout = out

	return &Process{Cmd: cmd, Start: time.Now(), Out: out}
}

func (p *Process) Run() {
	cmd := p.Cmd
	log.Println("=> exec  ", cmd.Path, cmd.Args)

	//outDir := JobOutputDir(job.URL)
	defer p.Out.Close()

	WLock()               // Cmd.Start() modifies state
	err1 := p.Cmd.Start() // err?
	WUnlock()

	err2 := p.Cmd.Wait()

	// TODO: determine proper status number
	if err1 != nil || err2 != nil {
		log.Println(cmd.Path, cmd.Args, err1, err2)
		return // TODO: write stat
	}
	return // TODO: write stat
}

//// TODO: revise
//func (n *Node) PeersWithJobs() []PeerInfo {
//	n.lock()
//	defer n.unlock()
//	var peers []PeerInfo
//	for _, p := range n.Peers {
//		//if p.HaveJobs {
//		peers = append(peers, p)
//		//}
//	}
//	return peers
//}
//
//type GPU struct {
//	Info string
//}
//

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
