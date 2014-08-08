package main

import "log"

type que struct {
	jobs []*Job
}

func (n *Node) GiveJob(nodeAddr string) string {
	return "a job for " + nodeAddr
}

func (l *que) Pop() *Job {
	jobs := l.jobs
	if len(jobs) == 0 {
		return nil
	}
	j := jobs[len(jobs)-1]
	jobs = jobs[:len(jobs)-1]
	return j
}

func (l *que) Push(j *Job) {
	log.Println("Push job:", j)
	l.jobs = append(l.jobs, j)
}

func (l *que) ListFiles() []string {
	files := make([]string, 0, len(l.jobs))
	for _, j := range l.jobs {
		files = append(files, j.fname)
	}
	return files
}
