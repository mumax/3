package main

/*
	Compute service runs jobs on this node's GPUs, if any
*/

import (
	"fmt"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"io"
	"log"
	"os/exec"
	"strings"
	"time"
)

var (
	MumaxVersion string
	GPUs         []string
	RunningHere  map[string]*Job
)

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
		URL := WaitForJob() // take an available job
		go func() {

			cmd, _ := makeProcess(URL, gpu, GUIAddr)
			//defer stdout.Close()

			// put job in "running" list (for status reporting)
			job := &Job{
				URL: URL,
				Cmd: cmd,
				//Node:   n.Addr,
				//GPU:    gpu,
				//Start:  time.Now(),
				//Status: RUNNING,
			}

			WLock()
			RunningHere[URL] = job
			WUnlock()

			_ = runJob(job)
			// TODO: write status
			//n.FSServer.CloseAll(JobOutputDir(JobInputFile(job.URL)))

			// remove from "running" list
			WLock()
			job.Cmd = nil
			delete(RunningHere, URL)
			//requeue := job.Reque > 0
			WUnlock()

			// notify job queue it's ready (for status)
			//if requeue {
			//	status = -1
			//}
			_, err := RPCCall(JobHost(URL), "NotifyJob", URL)
			if err != nil {
				log.Println(err)
			}

			// add GPU number back to idle stack
			idle <- gpu
		}()
	}
}

func WaitForJob() string {
	URL := FindJob()
	for URL == "" {
		time.Sleep(2 * time.Second) // TODO: don't poll
		URL = FindJob()
	}
	//log.Println("found job", URL)
	return URL
}

func FindJob() string {
	return "" // TODO
	//peers := PeersWithJobs()
	//for _, p := range peers {
	//	URL, err := n.RPCCall(p.Addr, "GiveJob", n.Addr)
	//	if err == nil && URL.(string) != "" {
	//		return URL.(string)
	//	}
	//}
	//return ""
}

//func (n *Node) KillJob(url string) error {
//	n.lock()
//	defer n.unlock()
//
//	log.Println("kill", url)
//	job := n.RunningHere[url]
//	if job == nil {
//		return fmt.Errorf("kill %v: job not running.", url)
//	}
//	job.Reque++
//	err := job.Cmd.Process.Kill()
//	return err
//}

// prepare exec.Cmd to run mumax3 compute process
func makeProcess(inputURL string, gpu int, webAddr string) (*exec.Cmd, io.WriteCloser) {
	// prepare command
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	forceFlag := `-f=0`
	cmd := exec.Command(command, gpuFlag, httpFlag, cacheFlag, forceFlag, inputURL)

	// Pipe stdout, stderr to log file over httpfs
	outDir := util.NoExt(JobInputFile(inputURL)) + ".out"
	httpfs.Mkdir(outDir)
	out, errD := httpfs.Create(outDir + "/stdout.txt")
	if errD != nil {
		log.Println(errD)
		return nil, nil
	}
	cmd.Stderr = out
	cmd.Stdout = out

	return cmd, out
}

func runJob(job *Job) (status int) {
	cmd := job.Cmd

	if cmd == nil {
		return 1
	}

	log.Println("=> exec  ", cmd.Path, cmd.Args)
	defer log.Println("<= exec  ", cmd.Path, cmd.Args, "status", status)

	//outDir := JobOutputDir(job.URL)
	err := cmd.Run()

	//defer cmd.Stdout.Close() // TODO: in closeall?

	// close all this simulation's files in case process exited ungracefully

	// TODO: determine proper status number
	if err != nil {
		log.Println(job.URL, err)
		return 1
	}
	return 0
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
