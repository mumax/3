package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"
)

var (
	flag_host  = flag.String("host", "", "override hostname")
	flag_poll  = flag.Duration("poll", 1*time.Second, "directory poll time")
	flag_relax = flag.Duration("relax", 1*time.Second, "relax time after job")
)

func main() {
	flag.Parse()
	rand.Seed(time.Now().UnixNano())
	lastdecay = time.Now()

	if flag.NArg() == 0 {
		log.Println("need args: directories to watch")
		os.Exit(1)
	}
	dirs := flag.Args()

	// share map stores how many seconds of compute time
	// was used by each que.
	share := make(map[string]float64)
	for _, d := range dirs {
		share[d] = rand.Float64()
	}

	//log.Println("inital share", share)

	if *flag_host == "" {
		h, err := os.Hostname()
		check(err)
		*flag_host = h
	}

	for {
		// find que with least share
		minshare := math.Inf(1)
		que, job, lock := "", "", ""
		for q, s := range share {
			j, l, ok := findJobFile(q)
			if ok && s < minshare {
				que = q
				job = j
				lock = l
				minshare = s
			}
		}

		if job != "" {
			start := time.Now()
			runJob(job, lock)
			seconds := time.Since(start).Seconds()
			share[que] += float64(seconds)
			decay(share)
			//log.Println("share", share)
		} else {
			time.Sleep(*flag_poll)
		}
		time.Sleep(*flag_relax)
	}

}

var (
	lastdecay   time.Time       // Last time the shares were decayed.
	decaytick   = 1 * time.Hour // Update decays every decaytick
	decayfactor = 0.997         // Multiply shares by this every decaytick, 0.997 gives half-life of about a week.
)

// update shares to decay by decayfactor/decaytick
func decay(share map[string]float64) {
	ok := false
	for time.Since(lastdecay) > decaytick {
		for k := range share {
			share[k] *= decayfactor
		}
		lastdecay = lastdecay.Add(decaytick)
		ok = true
	}
	if ok {
		log.Println("shares have decayed to", share)
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
	log.Println("exec", job.Command, job.Args)
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
func findJobFile(que string) (jobfile, lockfile string, ok bool) {
	dir, err := os.Open(que)
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
				return que + "/" + f, que + "/" + lockfile, true
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
