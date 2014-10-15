package main

import (
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

/*
Queue service scans the working directory for job files.
The working directory should contain per-user subdirectories. E.g.:
	arne/
	bartel/
	...
The in-memory representation is a cache and can be out-of-date at any point.

The queue service decides which job to hand out to a node if asked so.

*/

var (
	Users = make(map[string]*User) // maps user -> joblist
)

// RPC-callable method: picks a job of the queue returns it
// for the node to run it.
func GiveJob(nodeAddr string) string {
	WLock()
	defer WUnlock()
	user := nextUser()
	if user == "" {
		return ""
	}
	Users[user].FairShare += 1 // 1 second penalty because a job has started
	return Users[user].giveJob(nodeAddr).ID
}

func AddFairShare(s string) string {
	username := BaseDir(s)
	share := atoi(s[len(username)+1:])

	WLock()
	defer WUnlock()
	u := Users[username]
	if u == nil {
		return "no user " + username
	}
	log.Println("AddFairShare", username, share)
	u.FairShare += float64(share)
	return "" // ok
}

func nextUser() string {
	// search user with least share and jobs in queue
	leastShare := math.Inf(1)
	var bestUser string
	for n, u := range Users {
		if u.HasJob() && u.FairShare < leastShare {
			leastShare = u.FairShare
			bestUser = n
		}
	}
	return bestUser
}

// (Re-)load all jobs in the working directory.
// Called upon program startup.
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
func LoadUserJobs(dir string) string {
	log.Println("LoadUserJobs", dir)
	var newJobs []*Job
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if strings.HasSuffix(path, ".mx3") {
			ID := thisAddr + "/" + path
			job := &Job{ID: ID}
			job.Update()
			newJobs = append(newJobs, job)
		}
		return nil
	})
	l := joblist(newJobs)
	sort.Sort(&l)

	Fatal(err) // TODO: recover?

	WLock()
	defer WUnlock()
	if _, ok := Users[dir]; !ok {
		Users[dir] = NewUser()
	}
	Users[dir].Jobs = newJobs
	Users[dir].nextPtr = 0

	return ""
}

type joblist []*Job

func (l *joblist) Len() int           { return len(*l) }
func (l *joblist) Less(i, j int) bool { return (*l)[i].ID < (*l)[j].ID }
func (l *joblist) Swap(i, j int)      { (*l)[i], (*l)[j] = (*l)[j], (*l)[i] }

// RPC-callable function. Refreshes the in-memory cached info about this job.
// Called, e.g., after a node has finished a job.
func UpdateJob(jobURL string) string {

	WLock()
	defer WUnlock()

	j := JobByName(jobURL)
	if j == nil {
		log.Println("update", jobURL, ": no such job")
		return "" // empty conventionally means error
	}
	j.Update()

	return "updated " + jobURL // not used, but handy if called by Human.
}

// Periodically updates user's usedShare so they decay
// exponentially according to flag_haflife
func RunShareDecay() {
	halflife := *flag_halflife
	quantum := halflife / 100 // several updates per half-life gives smooth decay
	reduce := math.Pow(0.5, float64(quantum)/float64(halflife))
	for {
		time.Sleep(quantum)
		WLock()
		for _, u := range Users {
			u.FairShare *= reduce
		}
		WUnlock()
	}
}
