package main

import (
	"github.com/mumax/3/util"
	"net/url"
	"strings"
	"time"
)

type Job struct {
	File   string
	outDir string
	Node   string
	GPU    int
	Start  time.Time
	Stop   time.Time
	Status
}

type Status int

const (
	QUEUED Status = iota
	RUNNING
	FINISHED
	FAILED
)

var statusString = map[Status]string{
	QUEUED:   "QUEUED",
	RUNNING:  "RUNNING",
	FINISHED: "FINISHED",
	FAILED:   "FAILED",
}

func (s Status) String() string {
	return statusString[s]
}

func NewJob(file string) Job { return Job{File: file} }

func (j *Job) Runtime() time.Duration {
	if j.Stop.IsZero() {
		return since(time.Now(), j.Start)
	} else {
		return since(j.Stop, j.Start)
	}
}

func (j *Job) OutDir() string {
	if j.outDir == "" {
		j.outDir = util.NoExt(j.File) + ".out/"
	}
	return j.outDir
}

func (j *Job) HostName() string {
	colon := strings.Index(j.Node, ":")
	return j.Node[:colon]
}

func (j *Job) GUIPort() int {
	return GUI_PORT + j.GPU
}

func (j *Job) IsRunning() bool {
	return j.Status == RUNNING
}

func (j *Job) Failed() bool {
	return j.Status == FAILED
}

func JobHost(URL string) string {
	split := strings.Split(URL, "/")
	return split[2]
}

func JobUser(URL string) string {
	split := strings.Split(URL, "/")
	return split[4]
}

func JobInputFile(inputFile string) string {
	URL, err := url.Parse(inputFile)
	if err != nil {
		panic(err)
	}
	split := strings.Split(URL.Path, "/")
	if len(split) < 3 {
		panic("invalid url:" + inputFile)
	}
	baseHandler := "/" + split[1]
	return URL.Path[len(baseHandler):]
}
