package main

import "log"

type que struct {
	jobs    []Job
	running []Job
}

func (n *Node) GiveJob(nodeAddr string) string {
	n.lock()
	defer n.unlock()

	job := n.jobs.Pop()
	if job.File == "" {
		return "" // indicates no job available
	}
	job.Node = nodeAddr
	n.jobs.running = append(n.jobs.running, job)
	url := "http://" + n.inf.Addr + "/fs/" + job.File
	log.Println("give job", url, "->", nodeAddr)
	return url
}

func (l *que) Pop() Job {
	jobs := l.jobs
	if len(jobs) == 0 {
		return Job{}
	}
	j := jobs[len(jobs)-1]
	jobs = jobs[:len(jobs)-1]
	return j
}

func (l *que) Push(j Job) {
	log.Println("Push job:", j)
	l.jobs = append(l.jobs, j)
}

func (l *que) listQue() []Job {
	return copyJobs(l.jobs)
}

func (l *que) listRunning() []Job {
	return copyJobs(l.running)
}

// convert []*Job to []string (job names),
// for http status etc.
func copyJobs(jobs []Job) []Job {
	return append([]Job{}, jobs...)
}
