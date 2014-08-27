package main

import (
	"log"
	"math"
	"path"
	"strings"
	"time"
)

type User struct {
	Share                    float64         // Relative share of compute time
	Used                     float64         // Used-up compute time (decays)
	Queue, Running, Finished map[string]*Job // Queued, Running, Finished jobs
}

func (u *User) HasJob() bool {
	return len(u.Queue) > 0
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
	var job *Job

	// take random job from Queue map
	for _, j := range u.Queue {
		job = j
		break
	}
	delete(u.Queue, job.File)

	job.Status = RUNNING
	job.Start = time.Now()
	job.Node = nodeAddr
	job.Node = nodeAddr

	u.Running[job.File] = job

	log.Println("give job", job.File, "->", nodeAddr)
	return job.File
}

func (n *Node) NotifyJobFinished(jobURL string, status int) {
	log.Println("NotifyJobFinished", jobURL, status)
	username := JobUser(jobURL)
	user := n.Users[username]

	job := user.Running[jobURL]

	if status == 0 {
		job.Status = FINISHED
	} else {
		job.Status = FAILED
	}

	job.Stop = time.Now()
	delete(user.Running, jobURL)
	user.Finished[jobURL] = job
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
	if u.Queue == nil {
		u.Queue = make(map[string]*Job)
		u.Running = make(map[string]*Job)
		u.Finished = make(map[string]*Job)
	}
	url := "http://" + node.Addr + path.Clean("/fs/"+fname)
	u.Queue[url] = &Job{
		File: url,
	}
}

// convert []*Job to []string (job names),
// for http status etc.
func copyJobs(jobs []Job) []Job {
	return append([]Job{}, jobs...)
}
