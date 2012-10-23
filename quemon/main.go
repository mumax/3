package main

// Author: Mykola Dvornik
// Modified by Arne Vansteenkiste

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"sort"
	"strings"
	"time"
    "encoding/json"
    "log"
    "io"
)

var (
	home    = flag.String("home", "/cluster/home", "Home directory to look at")
	out     = flag.String("output", "", "HTML output directory")
	DirInfo []os.FileInfo
    STATUSMAP = map[int] string {
            -1: "FAILED",
             0: "FINISHED",
             1: "RUNNING",
             2: "PENDING",
    }
)

const (
	OUTNAME    = "status.html"
    STATUSFILENAME = "status.json"
	HTMLHeader = `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
body {
        background-color:rgb(250,250,250)
}

.failed {
        background-color:rgb(255,128,128)
}

.pending {
        background-color:rgb(255,255,128)
}

.finished {
        background-color:rgb(128,128,255);
        color:rgb(255,255,255)
}

.running {
        background-color:rgb(128,255,128)
}
</style>
</head>`
)

// Fields for Job.Status
const(
	Failed = -1
	Finished = 0
	Running = 1
)

type Job struct {
	Name      string
	Status    int // -1: failed, 0: finished, 1:running
	StartTime time.Time
	Runtime   time.Duration
	Node      string
	Gpu       int
}

var (
	filelist [][]string
	joblist  [][]Job
	users    []string
)

func GetListOfUsers() {
	info, _ := ioutil.ReadDir(*home)
	for i := range info {
		username := path.Base(info[i].Name())
		if username != "mumax" {
			users = append(users, username)
		}
	}
	sort.Strings(users)
}

func DumpListOfUsers() {
	for i := range users {
		log.Println(users[i])
	}

}

func GetDaemonFiles(userid int) {
	dir := *home + "/" + users[userid] + "/que/"
	_, err := os.Stat(dir)
	if os.IsNotExist(err) {
            log.Println(err)
		    return
	}
	DirInfo, _ = ioutil.ReadDir(dir)
	for i := range DirInfo {
		isExist, _ := path.Match("*.json", DirInfo[i].Name())
		if isExist {
			name := DirInfo[i].Name()
			filelist[userid] = append(filelist[userid], name)
		}
	}
}

func ProcessJobFromLog(userid int) {
	for i := range filelist[userid] {
		fname := filelist[userid][i]
		//check if json already has .status file
		filemask := fname[:len(fname)-len(path.Ext(fname))] + "*.out"
		var job Job

		var fileName string

        quePrefix := *home + "/" + users[userid] + "/que/"
        
		file := new(os.File)
		defer file.Close()
		var err error

		exist := false
		for j := range DirInfo {
            haveDir, _ := path.Match(filemask, DirInfo[j].Name())
			if haveDir {
                    dirname := DirInfo[j].Name()
                    //check if the status file is there
                    fileName = quePrefix + dirname + "/" + STATUSFILENAME;
                    _, err = os.Stat(fileName)
                    if os.IsNotExist(err) {
                            exist = false
                    } else {
                            exist = true
                    }
                    break
			}
		}
		if exist {
			file, err = os.Open(fileName)
			if err != nil {
				return
			}
            dec := json.NewDecoder(file)
            if err = dec.Decode(&job); err == io.EOF {
                    break
            } else if err != nil {
                    log.Print(err)
            }

		} else {
			job.Name = fname
			job.Status = 2
		}
		joblist[userid] = append(joblist[userid], job)
	}
}

func DumpQue() {
	fmt.Println("The que is following:")
	fmt.Println("UserName\tJob's name\tNode\tGPU\tStarted\tDuration\tStatus")
	for i := range users {
		UserName := users[i]
		for j := range joblist[i] {
			job := joblist[i][j]
			Node := job.Node
			GPU := fmt.Sprintf("%d", job.Gpu)
            StartTime := job.StartTime.Format("02 Jan 06 15:04")
			Duration := job.Runtime.String()
			fmt.Printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n", UserName, job.Name, Node, GPU, StartTime, Duration, STATUSMAP[job.Status])
		}
	}
}

func DumpQueHtml() {
	htmlFooter := "</html>\n"

	var body string
	body += fmt.Sprint(HTMLHeader)
	body += fmt.Sprintf("<title>%s</title>\n", "que status page")
    body += "</header>"
	body += fmt.Sprintf("<p>The status of the que as of %s\n\n</p><hr>", time.Now().String())
    tempBodyGlobal := ""
    
    var vStatusGlobal [4]int
    
	for i := range users {
	    
	    tempBodyUser := ""
	    var vStatusUser [4]int
	    
		UserName := users[i]
		tempBodyGlobal += fmt.Sprint("<b>" + UserName + "</b>\n")
		tempBodyGlobal += fmt.Sprint("<hr>\n")
		tempBodyUser += "<table>\n"

		tempBodyUser += fmt.Sprintf("<tr><b><td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</b>\n</tr>\n",
        "          ", "Job's name:", "Node name:", "GPU:", "Started on:", "Duration:", "Status:")
		for j := range joblist[i] {
			job := joblist[i][j]
			Node := job.Node
			GPU := fmt.Sprintf("%d", job.Gpu)
            StartTime := job.StartTime.Format("02 Jan 06 15:04:05")
			Duration := job.Runtime.String()
            Status := STATUSMAP[job.Status]
            vStatusUser[job.Status + 1] += 1
			tempBodyUser += fmt.Sprintf("<tr class='%s'>", strings.ToLower(Status))
			if job.Status != 2 {
				tempBodyUser += fmt.Sprintf("<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</tr>\n",
					"          ", job.Name, Node, GPU, StartTime, Duration, Status)
			} else {
				const None = "---"
				tempBodyUser += fmt.Sprintf("<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</tr>\n",
					"          ", job.Name, None, None, None, None, Status)
			}
		}
		tempBodyUser += "</table>\n"
		tempBodyGlobal += "<table cellpadding='10'>\n<tr>"
		for i := range vStatusUser {
		    Sts := STATUSMAP[i - 1];
		    vStatusGlobal[i] += vStatusUser[i]
		    tempBodyGlobal += fmt.Sprintf("<td class='%s'><b>%d</b></td>", strings.ToLower(Sts), vStatusUser[i])
		}
		tempBodyGlobal += "</tr></table>\n"
		tempBodyGlobal += tempBodyUser + "<hr>\n"
	}
	tempBodyGlobal += "</table>"
	body += fmt.Sprint("<b>Grand Total</b>\n")
	body += fmt.Sprint("<hr>\n")
	body += "<table cellpadding='10'>\n<tr>"
	for i := range vStatusGlobal {
	    Sts := STATUSMAP[i - 1];
	    body += fmt.Sprintf("<td class='%s'><b>%d</b></td>", strings.ToLower(Sts), vStatusGlobal[i])
	}
	body += "</tr></table>\n<hr>\n"
	body += tempBodyGlobal
	body += fmt.Sprint(htmlFooter)
	output := ([]byte)(body)
	filename := *out + "/" + OUTNAME
	file, err := os.OpenFile(filename, os.O_WRONLY | os.O_TRUNC | os.O_CREATE, 0666)
	defer file.Close()
	if err != nil {
		return
	}
	_, err1 := file.Write(output)
	if err1 != nil {
		log.Fatal(err1)
	}
}

func main() {

	flag.Parse()
	if len(*out) == 0 {
		*out, _ = os.Getwd()
	}
	_, err := os.Stat(*home)
	if os.IsNotExist(err) {
            log.Fatal(err)
    }
	GetListOfUsers()
	filelist = make([][]string, len(users))
	joblist = make([][]Job, len(users))

	for i := range users {
		GetDaemonFiles(i)
		ProcessJobFromLog(i)
	}
	DumpQueHtml()
}
