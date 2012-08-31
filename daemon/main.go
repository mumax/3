package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"
)

var (
	flag_dir   = flag.String("dir", ".", "directory to watch")
	flag_host  = flag.String("host", "", "override hostname")
	flag_poll  = flag.Duration("poll", 1*time.Second, "directory poll time")
	flag_relax = flag.Duration("relax", 1*time.Second, "relax time after job")
)

func main() {
	flag.Parse()

	if *flag_host == "" {
		h, err := os.Hostname()
		check(err)
		*flag_host = h
	}

	log.Println("watching", *flag_host, ":", *flag_dir)
	rand.Seed(time.Now().UnixNano())

	for {
		job, lock, ok := findJobFile()
		if ok {
			runJob(job, lock)
		} else {
			time.Sleep(*flag_poll)
		}
		time.Sleep(*flag_relax)
	}

}

// Run a job file.
func runJob(jobfile, lockdir string) {
	log.Println("starting", jobfile)

	err := os.Mkdir(lockdir, 0777)
	if err != nil {
		log.Println(err)
		return
	}

	j, err2 := os.Open(jobfile)
	if err2 != nil {
		log.Println(err2)
		return
	}
	defer j.Close()

	var job Job
	err3 := json.NewDecoder(j).Decode(&job)
	if err3 != nil {
		log.Println("error parsing", jobfile, ":", err3)
		return
	}
	spawn(job, lockdir)
}

// Spawn job subprocess.
func spawn(job Job, lockdir string) {
	cmd := exec.Command(job.Command, job.Args...)
	cmd.Dir = job.Wd

	// all daemon messages go here
	logout, err2 := os.OpenFile(lockdir+"/daemon.log",
		os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err2 != nil {
		log.Println(err2)
		return
	}
	defer logout.Close()
	fmt.Fprintln(logout, "this is the output of the daemon, your job's output is in stdout")

	// subprocess output goes here
	stdout, err3 := os.OpenFile(lockdir+"/stdout",
		os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err3 != nil {
		fmt.Fprintln(logout, err3)
		return
	}
	defer stdout.Close()
	cmd.Stdout = stdout
	cmd.Stderr = stdout

	// start
	fmt.Fprintln(logout, "job:", job)
	fmt.Fprintln(logout, "exec", job.Command, job.Args)
	fmt.Println("exec", job.Command, job.Args)
	err := cmd.Run()
	if err != nil {
		fmt.Fprintln(logout, err)
	}

	exitstat := "exited sucessfully"
	if !cmd.ProcessState.Success() {
		exitstat = "failed"
	}
	log.Println(job.Command, exitstat)
	fmt.Fprintln(logout, job.Command, exitstat)
}

// Find a job file that's not yet running.
func findJobFile() (jobfile, lockfile string, ok bool) {
	dir, err := os.Open(*flag_dir)
	check(err)
	defer dir.Close()
	files, err2 := dir.Readdirnames(-1)
	check(err2)

	// do not crash Intn():
	if len(files) == 0 {
		return
	}

	// start at random position, then go through files linearly
	start := rand.Intn(len(files))
	for I := range files {
		i := (I + start) % len(files)
		f := files[i]
		if path.Ext(f) == ".json" {
			if alreadyStarted(f, files) {
				continue
			} else {
				lockfile = noExt(f) + "_" + *flag_host + ".out"
				return f, lockfile, true
			}
		}
	}
	return "", "", false
}

// checks if a job is already started
// (.out exits)
func alreadyStarted(file string, files []string) bool {
	prefix := noExt(file)
	for _, f := range files {
		if strings.HasPrefix(f, prefix) && strings.HasSuffix(f, ".out") {
			//log.Println(file, "already started")
			return true
		}
	}
	//log.Println(file, "not yet started")
	return false
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// remove file extension.
func noExt(file string) string {
	return file[:len(file)-len(path.Ext(file))]
}
