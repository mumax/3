package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
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

func runJob(jobfile, lockfile string) {
	log.Println("starting", jobfile)
	lock, err := os.OpenFile(lockfile, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0777)
	if err != nil {
		log.Println("could not open", lockfile, ":", err)
		return
	}
	defer lock.Close()
}

func findJobFile() (jobfile, lockfile string, ok bool) {
	dir, err := os.Open(*flag_dir)
	check(err)
	defer dir.Close()
	files, err2 := dir.Readdirnames(-1)
	check(err2)

	// start at random position, then go through files linearly
	start := rand.Intn(len(files))
	for I := range files {
		i := (I + start) % len(files)
		f := files[i]
		if path.Ext(f) == ".json" {
			if alreadyStarted(f, files) {
				continue
			} else {
				lockfile = noExt(f) + "_" + *flag_host + ".lock"
				return f, lockfile, true
			}
		}
	}
	return "", "", false
}

func alreadyStarted(file string, files []string) bool {
	prefix := noExt(file)
	for _, f := range files {
		if strings.HasPrefix(f, prefix) && strings.HasSuffix(f, ".lock") {
			log.Println(file, "already started")
			return true
		}
	}
	log.Println(file, "not yet started")
	return false
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func noExt(file string) string {
	return file[:len(file)-len(path.Ext(file))]
}
