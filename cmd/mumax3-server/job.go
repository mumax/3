package main

import (
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/mumax/3/util"
)

// compute Job
type Job struct {
	URL string // URL of the input file, e.g., http://hostname/fs/user/inputfile.mx3
	// all of this is cache:
	Output  string // if exists, points to output url
	Engaged string // node address this job was last given to
	// old
	outputURL string    // URL of the output directory, access via OutputURL()
	Node      string    // Address of the node that runs/ran this job, if any. E.g.: computenode2:35360
	GPU       int       // GPU number on the compute node that runs/ran this job, if any
	Start     time.Time // When this job was started, if applicable
	Stop      time.Time // When this job was finished, if applicable
	Status              // Job status: queued, running,...
	Cmd       *exec.Cmd
	Reque     int // how many times requeued.
}

// read job files from storage and update status cache
func (j *Job) Update() {
	out := j.LocalOutputDir()
	if exists(out) {
		j.Output = out
	} else {
		j.Output = ""
	}
}

func JobByName(URL string) *Job {
	user := Users[BaseDir(LocalPath(URL))]
	if user == nil {
		log.Println("JobByName: no user for", URL)
		return nil
	}
	jobs := user.Jobs

	low := 0
	high := len(jobs) - 1
	mid := -1

	for low <= high {
		mid = (low + high) / 2
		switch {
		case jobs[mid].URL < URL:
			low = mid + 1
		case jobs[mid].URL > URL:
			high = mid + -1
		default:
			break
		}
	}

	if mid >= 0 && mid < len(jobs) && jobs[mid].URL == URL {
		return jobs[mid]
	} else {
		log.Println("JobByName: not found:", URL)
		return nil
	}
}

func (j *Job) User() string {
	return BaseDir(j.LocalPath())
}

// local path of input file
func (j *Job) LocalPath() string {
	return LocalPath(j.URL)
}

// local path of input file
func LocalPath(URL string) string {
	return MustParseURL(URL).Path[1:]
}

// local path of output dir
func (j *Job) LocalOutputDir() string {
	return util.NoExt(j.LocalPath()) + ".out/"
}

// insert "/fs" in front of url path
func (*Job) FS(url string) string {
	u := MustParseURL(url)
	u.Path = "/fs" + u.Path
	return u.String()
}

func (j *Job) IsQueued() bool {
	return j.Output == "" && j.Engaged == ""
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
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

//func (j *Job) Path() string {
//	return j.URL[len("http://"):]
//}

//func NewJob(URL string) Job {
//	return Job{URL: URL}
//}

// Returns how long this job has been running
func (j *Job) Runtime() time.Duration {
	if j.Start.IsZero() {
		return 0
	}
	if j.Stop.IsZero() {
		return Since(time.Now(), j.Start)
	} else {
		return Since(j.Stop, j.Start)
	}
}

// URL of the output directory.
func (j *Job) OutputURL() string {
	return util.NoExt(j.URL) + ".out"
}

//// Node host (w/o port) this job runs on, if any
//func (j *Job) NodeName() string {
//	colon := strings.Index(j.Node, ":")
//	if colon < 0 {
//		return ""
//	}
//	return j.Node[:colon]
//}

//func JobOutputDir(URL string) string {
//	return util.NoExt(URL) + ".out/"
//}

//func (j *Job) GUIPort() int {
//	return GUI_PORT + j.GPU
//}
//
//func (j *Job) IsRunning() bool {
//	return j.Status == RUNNING
//}
//
//func (j *Job) Failed() bool {
//	return j.Status == FAILED
//}
//
func JobHost(URL string) string {
	split := strings.Split(URL, "/")
	return split[2]
}

//func JobUser(URL string) string {
//	split := strings.Split(URL, "/")
//	return split[4]
//}
//
//func JobInputFile(inputFile string) string {
//	URL, err := url.Parse(inputFile)
//	if err != nil {
//		panic(err)
//	}
//	split := strings.Split(URL.Path, "/")
//	if len(split) < 3 {
//		panic("invalid url:" + inputFile)
//	}
//	baseHandler := "/" + split[1]
//	return URL.Path[len(baseHandler):]
//}
