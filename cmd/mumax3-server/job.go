package main

import (
	"github.com/mumax/3/util"
	"net/url"
	"os/exec"
	"strings"
	"time"
)

// compute Job
type Job struct {
	URL string // URL of the input file, e.g., http://hostname/fs/user/inputfile.mx3
	// all of this is cache:
	outputURL string    // URL of the output directory, access via OutputURL()
	Node      string    // Address of the node that runs/ran this job, if any. E.g.: computenode2:35360
	GPU       int       // GPU number on the compute node that runs/ran this job, if any
	Start     time.Time // When this job was started, if applicable
	Stop      time.Time // When this job was finished, if applicable
	Status              // Job status: queued, running,...
	Cmd       *exec.Cmd
	Reque     int // how many times requeued.
}

// Job status number queued, running,...
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

func (j *Job) Path() string {
	return j.URL[len("http://"):]
}

func NewJob(URL string) Job {
	return Job{URL: URL}
}

// Returns how long this job has been running
func (j *Job) Runtime() time.Duration {
	if j.Start.IsZero() {
		return 0
	}
	if j.Stop.IsZero() {
		return since(time.Now(), j.Start)
	} else {
		return since(j.Stop, j.Start)
	}
}

// URL of the output directory.
func (j *Job) OutputURL() string {
	if j.outputURL == "" {
		j.outputURL = JobOutputDir(j.URL)
	}
	return j.outputURL
}

// Node host (w/o port) this job runs on, if any
func (j *Job) NodeName() string {
	colon := strings.Index(j.Node, ":")
	if colon < 0 {
		return ""
	}
	return j.Node[:colon]
}

func JobOutputDir(URL string) string {
	return util.NoExt(URL) + ".out/"
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
