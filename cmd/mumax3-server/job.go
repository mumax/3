package main

import (
	"time"
)

type Job struct {
	File  string
	Node  string
	GPU   int
	Start time.Time
}

func NewJob(file string) Job { return Job{File: file} }

func (j Job) Runtime() time.Duration {
	return since(j.Start)
}
