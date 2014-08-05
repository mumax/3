package main

import "log"

type jobList struct {
	jobs []*Job
}

type Job struct {
	fname string
}

func NewJob(file string) *Job { return &Job{fname: file} }

func (l *jobList) Pop() *Job {
	jobs := l.jobs
	if len(jobs) == 0 {
		return nil
	}
	j := jobs[len(jobs)-1]
	jobs = jobs[:len(jobs)-1]
	return j
}

func (l *jobList) Push(j *Job) {
	log.Println("Push job:", j)
	l.jobs = append(l.jobs, j)
}

func (l *jobList) ListFiles() []string {
	files := make([]string, 0, len(l.jobs))
	for _, j := range l.jobs {
		files = append(files, j.fname)
	}
	return files
}
