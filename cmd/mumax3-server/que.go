package main

import (
	"log"
	"math"
	"path"
	"strings"
	"time"
)

type User struct {
	Share   float64 // Relative share of compute time
	Used    float64 // Used-up compute time (decays)
	Jobs    []*Job  // User's job queue
	NextJob int     // Points to next job to be run
}

func (u *User) HasJob() bool {
	return len(u.Jobs) > 0 && u.NextJob < len(u.Jobs)
}

// RPC-callable method: picks a job of the queue returns it
// for the node to run it.
func (n *Node) GiveJob(nodeAddr string) string {
	n.lock()
	defer n.unlock()

	// search user with least share and jobs in queue
	leastShare := math.Inf(1)
	var bestUser *User
	for _, u := range n.Users {
		if u.HasJob() && u.Used/u.Share < leastShare {
			leastShare = u.Used / u.Share
			bestUser = u
		}
	}
	if bestUser == nil {
		return ""
	}

	return bestUser.GiveJob(nodeAddr)
}

func (u *User) GiveJob(nodeAddr string) string {
	job := u.Jobs[u.NextJob]
	job.Status = RUNNING
	job.Start = time.Now()
	job.Node = nodeAddr
	u.NextJob++
	job.Node = nodeAddr
	url := "http://" + node.Addr + path.Clean("/fs/"+job.File)
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

func (u *User) AddJob(fname string) {
	u.Jobs = append(u.Jobs, &Job{
		File: fname,
	})
}

// convert []*Job to []string (job names),
// for http status etc.
func copyJobs(jobs []Job) []Job {
	return append([]Job{}, jobs...)
}
