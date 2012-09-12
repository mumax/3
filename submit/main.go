package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/user"
	"path"
	"strings"
)

func main() {
	flag.Parse()

	// get user
	u, err2 := user.Current()
	if err2 != nil {
		fmt.Fprintln(os.Stderr, err2)
		os.Exit(1)
	}

	// check if target is set
	if mkjob == nil {
		fmt.Fprintln(os.Stderr, "where would you like to submit to today?")
		os.Exit(1)
	}

	// set working directory
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	que := findque(wd)
	if que == "" {
		fmt.Fprintln(os.Stderr, "you're not in a place to submit files from")
		os.Exit(1)
	}
	wd = strip(wd, path.Dir(que))

	fmt.Println("submitting to", que)
	fmt.Println("using working directory $HOME/que/" + wd)

	// submit!
	for _, f := range flag.Args() {
		addJob(f, wd, que, u.Username)
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
func addJob(file, wd, que, usr string) {
	fmt.Println("add", file)
	job := mkjob(file)
	job.Wd = wd
	job.User = usr

	jobprefix := jsonfile(path.Clean(wd + "/" + noExt(file)))[1:]
	// remove previous .json and .out
	{
		dir, err := os.Open(que)
		if err != nil {
			fmt.Println(err)
			return
		}
		defer dir.Close()
		files, err2 := dir.Readdirnames(-1)
		if err2 != nil {
			fmt.Println(err2)
			return
		}

		prefix := noExt(jobprefix)
		for _, f := range files {
			if strings.HasPrefix(f, prefix) {
				fmt.Println("rm", que+"/"+f)
				err := os.RemoveAll(que + "/" + f)
				if err != nil {
					fmt.Println(err)
					return
				}
			}
		}
	}

	// add new json
	{
		jsonfile := que + "/" + jobprefix
		fmt.Println("add", jsonfile)
		out, err := os.OpenFile(jsonfile, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
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
	}
}

// suited job file name.
// replace / by _
func jsonfile(file string) string {
	return strings.Replace(file, "/", "_", -1) + ".json"
}

// remove file extension.
// TODO: -> lib
func noExt(file string) string {
	return file[:len(file)-len(path.Ext(file))]
}

func strip(wd, usr string) string {
	i := strings.Index(wd, usr)
	if i < 0 {
		fmt.Fprintln(os.Stderr, "working directory (", wd, ") should contain", usr)
		os.Exit(1)
	}
	return wd[i+len(usr):]
}
