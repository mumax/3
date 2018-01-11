package main

import (
	"log"
	"os"
	"strconv"
	"time"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

const MaxRequeue = 10 // maximum number of re-queues, don't run job if re-queued to many times

// compute Job
type Job struct {
	ID string // host/path of the input file, e.g., hostname:port/user/inputfile.mx3
	// in-memory properties:
	RequeCount int         // how many times requeued.
	Error      interface{} // error that cannot be consolidated to disk
	// all of this is cache:
	Output     string    // if exists, points to output ID
	Host       string    // node address in host file (=last host who started this job)
	ExitStatus string    // what's in the exitstatus file
	Start      time.Time // When this job was started, if applicable
	Alive      time.Time // Last time when this job was seen alive
	duration   time.Duration
}

// Find job belonging to ID
func JobByName(ID string) *Job {
	user := Users[BaseDir(LocalPath(ID))]
	if user == nil {
		log.Println("JobByName: no user for", ID)
		return nil
	}
	jobs := user.Jobs

	low := 0
	high := len(jobs) - 1
	mid := -1

	for low <= high {
		mid = (low + high) / 2
		switch {
		case jobs[mid].ID > ID:
			high = mid - 1
		case jobs[mid].ID < ID:
			low = mid + 1
		default:
			low = high + 1 // break for loop :-(
		}
	}

	if mid >= 0 && mid < len(jobs) && jobs[mid].ID == ID {
		return jobs[mid]
	} else {
		log.Println("JobByName: not found:", ID)
		return nil
	}
}

// read job files from storage and update status cache
func (j *Job) Update() {
	out := j.LocalOutputDir()
	if exists(out) {
		j.Output = thisAddr + "/" + out
	} else {
		j.Output = ""
		j.ExitStatus = ""
		j.Start = time.Time{}
		j.Alive = time.Time{}
		j.duration = 0
	}
	if j.Output != "" {
		j.Host = httpfsRead(out + "host")
		j.ExitStatus = httpfsRead(out + "exitstatus")
		j.Start = parseTime(httpfsRead(out + "start"))
		j.Alive = parseTime(httpfsRead(out + "alive"))
		j.duration = time.Duration(atoi(httpfsRead(out + "duration")))
	}
}

// Put job back in queue for later, e.g., when killed.
func (j *Job) Reque() {
	log.Println("requeue", j.ID)
	j.RequeCount++
	httpfs.Remove(j.LocalOutputDir())
	j.Update()
}

func SetJobError(ID string, err interface{}) {
	log.Println("SetJobErr", ID, err)
	WLock()
	defer WUnlock()
	j := JobByName(ID)
	if j == nil {
		return
	}
	j.Error = err
}

// How long job has been running, if running.
func (j *Job) Duration() time.Duration {
	if j.Start.IsZero() {
		return 0
	}
	if j.duration != 0 {
		return j.duration
	}
	if j.IsRunning() {
		return Since(time.Now(), j.Start)
	}
	return 0 // unknown duration
}

// user name for this job ID
func (j *Job) User() string {
	return JobUser(j.ID)
}

// user name for this job ID
func JobUser(ID string) string {
	return BaseDir(LocalPath(ID))
}

// local path of input file
func (j *Job) LocalPath() string {
	return LocalPath(j.ID)
}

// local path of input file, without host prefix. E.g.:
// 	host:123/user/file.mx3 -> user/file.mx3
func LocalPath(ID string) string {
	host := JobHost(ID)
	if len(host)+1 >= len(ID) {
		log.Println("Invalid LocalPath call on", ID)
		return ""
	}
	return ID[len(host)+1:]
}

// local path of output dir
func (j *Job) LocalOutputDir() string {
	return OutputDir(j.LocalPath())
}

// output directory for input file
func OutputDir(path string) string {
	return util.NoExt(path) + ".out/"
}

// insert "/fs" in front of url path
func (*Job) FS(id string) string {
	return FS(id)
}

// insert "/fs" in front of url path
func FS(id string) string {
	return BaseDir(id) + "/fs/" + LocalPath(id)
}

// is job queued?
func (j *Job) IsQueued() bool {
	return j.Output == "" && j.RequeCount < MaxRequeue
}

// is job running?
func (j *Job) IsRunning() bool {
	return j.Output != "" && j.ExitStatus == "" && j.Host!=""
}

// Host of job with this ID (=first path element). E.g.:
// 	host:123/user/file.mx3 -> host:123
func JobHost(ID string) string {
	return BaseDir(ID)
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

// human-readable status string (for gui)
func (j *Job) Status() string {
	if j.IsQueued() {
		return QUEUED.String()
	}
	if j.ExitStatus == "0" {
		return FINISHED.String()
	}
	if j.ExitStatus == "" && j.Host == "" {
		return FINISHED.String()
	}
	if j.Host != "" && j.ExitStatus == "" {
		return RUNNING.String()
	}
	if j.ExitStatus != "" && j.ExitStatus != "0" {
		return FAILED.String()
	}
	return "UNKNOWN"
}

// remove job output
func Rm(URL string) string {
	err := httpfs.Remove("http://" + OutputDir(URL))

	// update status after output removal
	UpdateJob(URL)

	if err != nil {
		return err.Error()
	}

	// report re-queue
	// handy if others remove your jobs
	job := JobByName(URL)
	if job != nil {
		job.RequeCount++
	}

	// make sure job runs again quickly
	user := JobUser(URL)
	u := Users[user]
	if u != nil {
		u.nextPtr = 0
	}
	return ""
}

// check if path exists
func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// atoi, does not return error
func atoi(a string) int64 {
	i, _ := strconv.ParseInt(a, 10, 64)
	return i
}

// return file content as string, no errors
func httpfsRead(fname string) string {
	data, err := httpfs.Read(fname)
	if err != nil {
		return ""
	}
	return string(data)
}
