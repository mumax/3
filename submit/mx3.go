package main

import (
	"flag"
)

var flag_mx3 = flag.Bool("mx3", false, "submit to mx3")

func init() {
	flag.Parse()
	if *flag_mx3 {
		setMkjob(mx3)
	}
}

func mx3(file string) Job {
	var job Job
	job.Command = "mx3"
	job.Args = []string{`-f`, `-gpu=%GPU`, file}
	return job
}
