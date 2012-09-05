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
	wd = strip(wd, u.Username)

	fmt.Println("submitting to", que)
	fmt.Println("using working directory $HOME/" + wd)

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
	fmt.Println("add", file)
	job := mkjob(file)
	job.Wd = wd

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

		prefix := noExt(jsonfile(file))
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
		fmt.Println("add", que+"/"+jsonfile(file))
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
	usr += "/"
	i := strings.Index(wd, usr)
	if i < 0 {
		fmt.Fprintln(os.Stderr, "working directory (", wd, ") should contain", usr)
		os.Exit(1)
	}
	return wd[i+len(usr):]
}
