package main

import (
	"fmt"
	"log"
	"os/exec"
	"time"
)

func (n *Node) RunComputeService() {
	idle := make(chan int, len(n.GPUs))
	for i := range n.GPUs {
		idle <- i
	}

	for {
		gpu := <-idle
		addr := fmt.Sprint(":", 35367+gpu)
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
	log.Println("found job", URL)
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

func run(inFile string, gpu int, webAddr string) int {
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	cmd := exec.Command(command, gpuFlag, httpFlag, inFile)
	//cmd.Stderr = os.Stderr
	//cmd.Stdout = os.Stdout
	//cmd.Stdin = os.Stdin
	log.Println(command, cacheFlag, gpuFlag, httpFlag, inFile)
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
