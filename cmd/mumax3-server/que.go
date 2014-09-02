package main

import (
	"math"
	"path"
	"strings"
	"time"
)

// RPC-callable method: picks a job of the queue returns it
// for the node to run it.
func (n *Node) GiveJob(nodeAddr string) string {
	n.lock()
	defer n.unlock()

	// search user with least share and jobs in queue
	leastShare := math.Inf(1)
	var bestUser *User
	for _, u := range n.Users {
		used := u.UsedShare()
		if u.HasJob() && used/u.Share < leastShare {
			leastShare = used / u.Share
			bestUser = u
		}
	}
	if bestUser == nil {
		return ""
	}

	return bestUser.GiveJob(nodeAddr)
}

func (n *Node) NotifyJobFinished(jobURL string, status int) {
	//log.Println("NotifyJobFinished", jobURL, status)

	n.lock()
	defer n.unlock()

	username := JobUser(jobURL)
	user := n.Users[username]

	job := user.Running[jobURL]

	if status == 0 {
		job.Status = FINISHED
	} else {
		job.Status = FAILED
	}

	job.Stop = time.Now()
	user.usedShare += job.Runtime().Seconds()
	delete(user.Running, jobURL)
	user.Finished[jobURL] = job
}

func (n *Node) AddJob(fname string) {
	n.lock()
	defer n.unlock()
	//log.Println("Push job:", fname)

	if path.IsAbs(fname) {
		if !strings.HasPrefix(fname, n.RootDir) {
			panic("AddJob " + fname + ": not in root: " + n.RootDir) // TODO: handle gracefully
		}
		fname = fname[len(n.RootDir):] // strip root prefix
	}

	split := strings.Split(fname, "/")
	first := ""
	for _, s := range split {
		if s != "" {
			first = s
			break
		}
	}

	if n.Users[first] == nil {
		n.Users[first] = &User{Share: 1}
	}

	n.Users[first].AddJob(fname)
}

// convert []*Job to []string (job names),
// for http status etc.
func copyJobs(jobs []Job) []Job {
	return append([]Job{}, jobs...)
}
