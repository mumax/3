package main

/*
import (
	"path"
	"time"
)

type Share struct {
	total               float64         // Relative share of compute time
	used                float64         // Used-up compute time in the past (decays)
}

// used share in GPU*hours
func (u *Share) Used() float64 {
	used := u.used
	for _, j := range u.Running {
		used += j.Runtime().Seconds()
	}
	return used / 3600
}

func (u *User) HasJob() bool {
	return len(u.Queue) > 0
}

func (u *User) GiveJob(nodeAddr string) string {
	var job *Job

	// take random job from Queue map
	for _, j := range u.Queue {
		job = j
		break
	}
	delete(u.Queue, job.URL)

	job.Status = RUNNING
	job.Start = time.Now()
	job.Node = nodeAddr
	job.Node = nodeAddr

	u.Running[job.URL] = job

	//log.Println("give job", job.URL, "->", nodeAddr)
	return job.URL
}

func (u *User) AddJob(fname string) {
	if u.Queue == nil {
		u.Queue = make(map[string]*Job)
		u.Running = make(map[string]*Job)
		u.Finished = make(map[string]*Job)
	}
	url := "http://" + node.Addr + path.Clean("/fs/"+fname)
	u.Queue[url] = &Job{
		URL: url,
	}
}
*/
