package main

import (
	"flag"
)

var flag_mumax2 = flag.Bool("mumax2", false, "submit to mumax2")

func init() {
	flag.Parse()
	if *flag_mumax2 {
		setMkjob(mumax2)
	}
}

func mumax2(file string) Job {
	var job Job
	job.Command = "mumax2"
	job.Args = []string{`-s`, `-f`, file}
	return job
}
