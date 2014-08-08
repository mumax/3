package main

type Job struct {
	File string
	Node string
}

func NewJob(file string) Job { return Job{File: file} }
