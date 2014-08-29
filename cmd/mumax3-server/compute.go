package main

import (
	"fmt"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"log"
	"os/exec"
	"strings"
	"time"
)

// Runs a compute service on this node, if GPUs are available.
// The compute service asks storage nodes for a job, runs it,
// saves results over httpfs and notifies storage when ready.
func (n *Node) RunComputeService() {
	if len(n.GPUs) == 0 {
		return
	}
	idle := make(chan int, len(n.GPUs))
	for i := range n.GPUs {
		idle <- i
	}

	for {
		gpu := <-idle
		addr := fmt.Sprint(":", GUI_PORT+gpu)
		URL := n.WaitForJob()
		go func() {
			n.lock()

			n.RunningHere[URL] = &Job{
				File:   URL,
				Node:   n.Addr,
				GPU:    gpu,
				Start:  time.Now(),
				Status: RUNNING,
			}
			n.unlock()

			status := run(URL, gpu, addr)

			n.lock()
			delete(n.RunningHere, URL)
			n.unlock()

			_, err := n.RPCCall(JobHost(URL), "NotifyJobFinished", URL, status)
			if err != nil {
				log.Println(err)
			}

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

// run input file and return status number
func run(inFile string, gpu int, webAddr string) (status int) {
	// prepare command
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	forceFlag := `-f=0`
	cmd := exec.Command(command, gpuFlag, httpFlag, cacheFlag, forceFlag, inFile)

	// Pipe stdout, stderr to log file over httpfs
	fs, errFS := httpfs.Dial("http://" + JobHost(inFile) + "/fs")
	if errFS != nil {
		log.Println(errFS)
		return -1
	}
	outDir := util.NoExt(JobInputFile(inFile)) + ".out"
	fs.Mkdir(outDir, 0777)
	out, errD := fs.Create(outDir + "/stdout.txt")
	if errD != nil {
		log.Println(errD)
		return -1
	}
	defer out.Close()
	cmd.Stderr = out
	cmd.Stdout = out

	log.Println("=> exec  ", cmd.Path, cmd.Args)
	defer log.Println("<= exec  ", cmd.Path, cmd.Args, "status", status)
	err := cmd.Run()
	// TODO: determine proper status number
	if err != nil {
		log.Println(inFile, err)
		return 1
	}
	return 0
}

func (n *Node) PeersWithJobs() []PeerInfo {
	n.lock()
	defer n.unlock()
	var peers []PeerInfo
	for _, p := range n.Peers {
		if p.HaveJobs {
			peers = append(peers, p)
		}
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
