package main

import (
	"log"
	"path"
	"strings"
)

// RPC-callable method: picks a job of the queue returns it
// for the node to run it.
func (n *Node) GiveJob(nodeAddr string) string {
	n.lock()
	defer n.unlock()

	if len(n.jobs) == 0 {
		return "" // indicates no job available
	}
	job := n.jobs[len(n.jobs)-1]
	n.jobs = n.jobs[:len(n.jobs)-1]

	job.Node = nodeAddr
	n.running = append(n.running, job)
	url := "http://" + n.Addr + path.Clean("/fs/"+job.File)
	log.Println("give job", url, "->", nodeAddr)
	return url
}

func (n *Node) AddJob(fname string) {
	n.lock()
	defer n.unlock()
	log.Println("Push job:", fname)

	if path.IsAbs(fname) {
		if !strings.HasPrefix(fname, n.RootDir) {
			panic("AddJob " + fname + ": not in root: " + n.RootDir) // TODO: handle gracefully
		}
		fname = fname[len(n.RootDir):] // strip root prefix
	}
	n.jobs = append(n.jobs, Job{File: fname})
}

// convert []*Job to []string (job names),
// for http status etc.
func copyJobs(jobs []Job) []Job {
	return append([]Job{}, jobs...)
}
