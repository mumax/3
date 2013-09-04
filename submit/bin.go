package main

import (
	"flag"
)

var flag_bin = flag.Bool("bin", false, "submit binary")

func init() {
	flag.Parse()
	if *flag_bin {
		setMkjob(bin)
	}
}

func bin(file string) Job {
	var job Job
	job.Command = "./" + file
	job.Args = []string{`-f`, `-gpu=%GPU`, `-sched=yield`}
	return job
}
