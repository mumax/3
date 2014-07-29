package main

import (
	"flag"
)

var flag_hotspin = flag.Bool("hotspin", false, "submit to hotspin")

func init() {
	flag.Parse()
	if *flag_hotspin {
		setMkjob(hotspin)
	}
}

func hotspin(file string) Job {
	var job Job
	job.Command = "hotspin"
	job.Args = []string{`-dr`, `-gpu=%GPU`, `-sched=yield`, file}
	return job
}
