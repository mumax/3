package main

import (
	"flag"
)

var flag_mumax3 = flag.Bool("mumax3", false, "submit to mumax3")

func init() {
	flag.Parse()
	if *flag_mumax3 {
		setMkjob(mumax3)
	}
}

func mumax3(file string) Job {
	var job Job
	job.Command = "mumax3"
	job.Args = []string{`-gpu=%GPU`, file}
	return job
}
