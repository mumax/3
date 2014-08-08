package main

type Job struct {
	fname string
}

func NewJob(file string) *Job { return &Job{fname: file} }
