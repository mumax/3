package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/user"
	"path"
)

var (
	flag_que = flag.String("que", "", "override default queue directory ($HOME/que)")
)

func main() {
	flag.Parse()

	// get user
	u, err2 := user.Current()
	if err2 != nil {
		fmt.Fprintln(os.Stderr, err2)
		os.Exit(1)
	}
	que := path.Clean(u.HomeDir + "/que")

	// override que directory by env.
	override := os.Getenv("MUMAX_QUE")
	if override != "" {
		que = override
	}

	// override que directory by cli flag
	if *flag_que != "" {
		que = *flag_que
	}

	// check if target is set
	if mkjob == nil {
		fmt.Fprintln(os.Stderr, "no target specified")
		os.Exit(1)
	}

	// set working directory
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Println("submitting to", que)
	fmt.Println("using working directory", wd)

	// submit!
	for _, f := range flag.Args() {
		addJob(f, wd, que)
	}
}

// Function used to make jobs.
// Set by one of the plug-ins.
// mkjob does not need to set the job's WD.
var mkjob func(file string) Job

// Sets mkjob, fails if already set.
func setMkjob(f func(string) Job) {
	if mkjob != nil {
		fmt.Fprintln(os.Stderr, "more than one target specified")
		os.Exit(1)
	}
	mkjob = f
}

// Add a job to the que directoy,
// using the mkjob function.
func addJob(file, wd, que string) {
	fmt.Print(file, ": ")
	job := mkjob(file)
	job.Wd = wd

	out, err := os.OpenFile(que+"/"+jsonfile(file), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer out.Close()

	err2 := json.NewEncoder(out).Encode(job)
	if err2 != nil {
		fmt.Println(err2)
		return
	}

	fmt.Println("OK")
}

func jsonfile(file string) string {
	return file + ".json"
}
