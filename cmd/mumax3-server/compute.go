package main

import (
	"fmt"
	"log"
	"os/exec"
	"strings"
	"time"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

// Runs a compute service on this node, if GPUs are available.
// The compute service asks storage nodes for a job, runs it,
// saves results over httpfs and notifies storage when ready.
func (n *Node) RunComputeService() {

	if len(n.GPUs) == 0 {
		return
	}

	// stack of available GPU numbers
	idle := make(chan int, len(n.GPUs))
	for i := range n.GPUs {
		idle <- i
	}

	for {
		gpu := <-idle // take an available GPU
		GUIAddr := fmt.Sprint(":", GUI_PORT+gpu)
		URL := n.WaitForJob() // take an available job
		go func() {
			// put job in "running" list (for status reporting)
			job := &Job{
				URL:    URL,
				Node:   n.Addr,
				GPU:    gpu,
				Start:  time.Now(),
				Status: RUNNING,
				Cmd:    makeProcess(URL, gpu, GUIAddr),
			}

			n.lock()
			n.RunningHere[URL] = job
			n.unlock()

			status := runJob(job)

			// remove from "running" list
			n.lock()
			job.Cmd = nil
			delete(n.RunningHere, URL)
			n.unlock()

			// notify job queue it's ready (for status)
			_, err := n.RPCCall(JobHost(URL), "NotifyJobFinished", URL, status)
			if err != nil {
				log.Println(err)
			}

			// add GPU number back to idle stack
			idle <- gpu
		}()
	}
}

func (n *Node) WaitForJob() string {
	URL := n.FindJob()
	for URL == "" {
		time.Sleep(2 * time.Second) // TODO: don't poll
		URL = n.FindJob()
	}
	//log.Println("found job", URL)
	return URL
}

func (n *Node) FindJob() string {
	peers := n.PeersWithJobs()
	for _, p := range peers {
		URL, err := n.RPCCall(p.Addr, "GiveJob", n.Addr)
		if err == nil && URL.(string) != "" {
			return URL.(string)
		}
	}
	return ""
}

func (n *Node) KillJob(url string) error {
	n.lock()
	defer n.unlock()

	log.Println("kill", url)
	job := n.RunningHere[url]
	if job == nil {
		return fmt.Errorf("kill %v: job not running.", url)
	}
	err := job.Cmd.Process.Kill()
	return err
}

// run input file at URL on gpu number, serve GUI at wabAddr and return status number
func makeProcess(inputURL string, gpu int, webAddr string) *exec.Cmd {
	// prepare command
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	forceFlag := `-f=0`
	cmd := exec.Command(command, gpuFlag, httpFlag, cacheFlag, forceFlag, inputURL)

	// Pipe stdout, stderr to log file over httpfs
	fs, errFS := httpfs.Dial("http://" + JobHost(inputURL) + "/fs")
	if errFS != nil {
		log.Println(errFS)
		return nil
	}
	outDir := util.NoExt(JobInputFile(inputURL)) + ".out"
	fs.Mkdir(outDir, 0777)
	out, errD := fs.Create(outDir + "/stdout.txt")
	if errD != nil {
		log.Println(errD)
		return nil
	}
	cmd.Stderr = out
	cmd.Stdout = out

	return cmd
}

func runJob(job *Job) (status int) {
	cmd := job.Cmd

	if cmd == nil {
		return -1
	}

	log.Println("=> exec  ", cmd.Path, cmd.Args)
	defer log.Println("<= exec  ", cmd.Path, cmd.Args, "status", status)

	outDir := JobOutputDir(job.URL)
	defer node.FSServer.CloseAll(outDir) // TODO: doesn't work because of http://??
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

// TODO: revise
func (n *Node) PeersWithJobs() []PeerInfo {
	n.lock()
	defer n.unlock()
	var peers []PeerInfo
	for _, p := range n.Peers {
		//if p.HaveJobs {
		peers = append(peers, p)
		//}
	}
	return peers
}

type GPU struct {
	Info string
}

const MAXGPU = 16

func DetectGPUs() []GPU {
	var gpus []GPU
	for i := 0; i < MAXGPU; i++ {
		gpuflag := fmt.Sprint("-gpu=", i)
		out, err := exec.Command(*flag_mumax, "-test", gpuflag).Output()
		if err == nil {
			info := string(out)
			if strings.HasSuffix(info, "\n") {
				info = info[:len(info)-1]
			}
			log.Println("gpu", i, ":", info)
			gpus = append(gpus, GPU{info})
		}
	}
	return gpus
}

func DetectMumax() string {
	out, err := exec.Command(*flag_mumax, "-test", "-v").Output()
	if err == nil {
		info := string(out)
		split := strings.SplitN(info, "\n", 2)
		version := split[0]
		log.Println("have", version)
		return version
	}
	return ""
}
