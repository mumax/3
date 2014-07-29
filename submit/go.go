package main

import (
	"flag"
)

var flag_go = flag.Bool("go", false, "submit go source")

func init() {
	flag.Parse()
	if *flag_go {
		setMkjob(go_)
	}
}

func go_(file string) Job {
	var job Job
	job.Command = "go"
	job.Args = []string{"run", file, `-gpu=%GPU`}
	return job
}
