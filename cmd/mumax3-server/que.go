package main

import (
	"log"
	"os"
	"path/filepath"
	"strings"
)

/*
Queue service scans the working directory for job files,
decides which job to hand out to free compute node.

The working directory should contain per-user subdirectories. E.g.:
	arne/
	bartel/
	...

*/

var (
	Jobs = make(map[string][]*Job) // maps user -> joblist
	//Users  map[string]*User
)

// (Re-)load all jobs in the working directory.
func LoadJobs() {
	dir, err := os.Open(".")
	Fatal(err)
	subdirs, err2 := dir.Readdir(-1)
	Fatal(err2)

	for _, d := range subdirs {
		if d.IsDir() {
			LoadUserJobs(d.Name())
		}
	}
}

// (Re-)load all jobs in the user's subdirectory.
func LoadUserJobs(dir string) {
	log.Println("LoadUserJobs", dir)
	var newJobs []*Job
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if strings.HasSuffix(path, ".mx3") {
			URL := "http://" + thisAddr + "/" + path
			newJobs = append(newJobs, &Job{URL: URL})
		}
		return nil
	})
	Fatal(err)
	WLock()
	defer WUnlock()
	Jobs[dir] = newJobs
}

// RPC-callable method: picks a job of the queue returns it
// for the node to run it.
func (*RPC) GiveJob(nodeAddr string) string {
	return ""
	// search user with least share and jobs in queue
	//	leastShare := math.Inf(1)
	//	var bestUser *User
	//	for _, u := range n.Users {
	//		used := u.UsedShare()
	//		if u.HasJob() && used/u.Share < leastShare {
	//			leastShare = used / u.Share
	//			bestUser = u
	//		}
	//	}
	//	if bestUser == nil {
	//		return ""
	//	}

	//return giveJob(bestUser, nodeAddr)
}

func (*RPC) NotifyJobFinished(jobURL string) {
	////log.Println("NotifyJobFinished", jobURL, status)

	//n.lock()
	//defer n.unlock()

	//username := JobUser(jobURL)
	//user := n.Users[username]

	//job := user.Running[jobURL]

	//// TODO: job == nil?

	//switch {
	//case status > 0:
	//	job.Status = FAILED
	//case status == 0:
	//	job.Status = FINISHED
	//case status < 0:
	//	job.Status = QUEUED
	//	job.Reque++
	//	job.Start = time.Time{}
	//	job.Stop = time.Time{}
	//}

	//// TODO: rm .out on requeue
	//job.Stop = time.Now()
	//if job.Status != QUEUED {
	//	user.usedShare += job.Runtime().Seconds()
	//}

	//delete(user.Running, jobURL)

	//if job.Status == FINISHED || job.Status == FAILED {
	//	user.Finished[jobURL] = job
	//}
	//if job.Status == QUEUED {
	//	user.Queue[jobURL] = job
	//}
}

// Periodically updates user's usedShare so they decay
// exponentially according to flag_haflife
//func RunShareDecay() {
//	halflife := *flag_halflife
//	quantum := halflife / 100 // several updates per half-life gives smooth decay
//	reduce := math.Pow(0.5, float64(quantum)/float64(halflife))
//	for {
//		time.Sleep(quantum)
//		node.lock()
//		for _, u := range node.Users {
//			u.usedShare *= reduce
//		}
//		node.unlock()
//	}
//}

//
//var scan = make(chan struct{})
//
//// RPC-callable method: picks a job of the queue returns it
//// for the node to run it.
//func (n *Node) ReScan() {
//	select {
//	default: // already scannning
//	case scan <- struct{}{}: // wake-up scanner
//	}
//}
//
//func exist(filename string) bool {
//	_, err := os.Stat(filename)
//	return err == nil
//}
