package main

import (
	"github.com/mumax/3/util"
	"time"
)

type Job struct {
	File   string
	outDir string
	Node   string
	GPU    int
	Start  time.Time
	Stop   time.Time
	Status
}

type Status int

const (
	QUEUED Status = iota
	RUNNING
	FINISHED
	FAILED
)

var statusString = map[Status]string{
	QUEUED:   "QUEUED",
	RUNNING:  "RUNNING",
	FINISHED: "FINISHED",
	FAILED:   "FAILED",
}

func (s Status) String() string {
	return statusString[s]
}

func NewJob(file string) Job { return Job{File: file} }

func (j *Job) Runtime() time.Duration {
	if j.Stop.IsZero() {
		return since(time.Now(), j.Start)
	} else {
		return since(j.Stop, j.Start)
	}
}

func (j *Job) OutDir() string {
	if j.outDir == "" {
		j.outDir = util.NoExt(j.File) + ".out/"
	}
	return j.outDir
}
