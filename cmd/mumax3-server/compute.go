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
			run(URL, gpu, addr)
			// add job to finished, report, etc
			idle <- gpu
		}()
	}
}

func (n *Node) WaitForJob() string {
	URL := n.FindJob()
	for URL == "" {
		time.Sleep(1 * time.Second) // TODO: don't poll
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

func run(inFile string, gpu int, webAddr string) {
	command := *flag_mumax
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)
	cacheFlag := fmt.Sprint(`-cache=`, *flag_cachedir)
	cmd := exec.Command(command, gpuFlag, httpFlag, inFile)
	log.Println(command, cacheFlag, gpuFlag, httpFlag, inFile)
	err := cmd.Run()
	if err != nil {
		log.Println(inFile, err)
		// TODO: report back
	}
}

func (n *Node) PeersWithJobs() []PeerInfo {
	n.lock()
	defer n.unlock()
	var peers []PeerInfo
	for _, p := range n.peers {
		if p.HaveJobs {
			peers = append(peers, p)
		}
	}
	return peers
}
