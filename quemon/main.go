package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"sort"
	"strings"
	"time"
)

var home = flag.String("home", "/cluster/home", "Home directory to look at")
var out = flag.String("output", "", "HTML output directory")
var DirInfo []os.FileInfo

const OUTNAME = "status.html"
const HTMLHeader = "<!DOCTYPE html>\n<html>\n<head>\n<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\">\n<style type=\"text/css\">\nbody {\n        background-color:rgb(250,250,250)\n}\n\n.failed {\n        background-color:rgb(255,128,128)\n}\n\n.pending {\n        background-color:rgb(255,255,128)\n}\n\n.finished {\n        background-color:rgb(128,128,255);\n        color:rgb(255,255,255)\n}\n\n.running {\n        background-color:rgb(128,255,128)\n}\n</style>\n</head>\n"

type Job struct {
	Name      string
	Status    int
	StartTime time.Time
	Runtime   time.Duration
	Node      int
	Gpu       int
}

var filelist [][]string
var joblist [][]Job
var users []string

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
		fmt.Printf("%s\n", users[i])
	}

}

func GetDaemonFiles(userid int) {
	dir := *home + "/" + users[userid] + "/que/"
	//fmt.Printf("Look into %s ...",dir)
	_, err := os.Stat(dir)
	if os.IsNotExist(err) {
		return
	}
	//fmt.Print("Proceeding...\n")
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
		//check if json already has .out dir
		filemask := fname[:len(fname)-len(path.Ext(fname))] + "*.out"

		status := 1
		var job Job
		var filename string
		file := new(os.File)
		defer file.Close()
		var err error

		exist := false
		for j := range DirInfo {
			exist, _ = path.Match(filemask, DirInfo[j].Name())
			if exist {
				filename = DirInfo[j].Name()
				break
			}
		}
		if exist {
			filename = *home + "/" + users[userid] + "/que/" + filename + "/daemon.log"
			file, err = os.Open(filename)
			if err != nil {
				return
			}
			buffer, err1 := ioutil.ReadAll(file)
			if err1 != nil {
				return
			}
			StrBuffer := (string)(buffer)

			fi, _ := file.Stat()
			job.StartTime = fi.ModTime()
			ByStrings := strings.Split(StrBuffer, "\n")
			for i := range ByStrings {
				if strings.Contains(ByStrings[i], "exec") {
					job.Name = ByStrings[i][strings.Index(ByStrings[i], "exec")+len("exec "):]
					break
				}
			}
			//Get status
			if strings.Contains(StrBuffer, "fail") {
				status = -1 // -1 - job failed
			}
			if strings.Contains(StrBuffer, "sucessfully") {
				status = 0
			}
			if status == 1 {
				job.Runtime = time.Now().Sub(job.StartTime)
			}

			job.Status = status
			//Guess node
			indexOfNode := strings.Index(filename, "node")
			if indexOfNode < 0 {
				return
			}
			nodeName := filename[indexOfNode+len("node"):]
			nodeName = nodeName[:strings.Index(nodeName, ".")]
			// nodeName is in XXXX_X format
			nodeNameFmt := strings.Replace(nodeName, "_", "\t", -1)
			//nodeNameFmt = strings.TrimLeft(nodeNameFmt, "0")
			//fmt.Printf("%s\n", nodeNameFmt)
			fmt.Sscanf(nodeNameFmt+"\n", "%v %v", &job.Node, &job.Gpu)
		} else {
			job.Name = fname
			job.Status = 2
		}
		joblist[userid] = append(joblist[userid], job)
	}
}

func DumpQue() {
	fmt.Print("The que is following:\n")
	fmt.Print("UserName\tJob's name\tNode\tGPU\tStarted\tDuration\tStatus\n")
	for i := range users {
		UserName := users[i]
		for j := range joblist[i] {
			job := joblist[i][j]
			Node := fmt.Sprintf("%02d", job.Node)
			GPU := fmt.Sprintf("%d", job.Gpu)
			StartTime := job.StartTime.Format("01-Sep-2006 15:04:05")
			Duration := job.Runtime.String()
			var Status string
			switch job.Status {
			case -1:
				Status = "FAILED"
			case 0:
				Status = "FINISHED"
			case 1:
				Status = "RUNNING"
			case 2:
				Status = "PENDING"
			}
			fmt.Printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n", UserName, job.Name, Node, GPU, StartTime, Duration, Status)
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

	for i := range users {
		UserName := users[i]
		body += fmt.Sprint("<b>" + UserName + "</b>\n")
		body += fmt.Sprint("<hr>\n")
		body += "<table>\n"

		body += fmt.Sprintf("<tr><b><td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</b>\n</tr>\n",
			"          ", "Job's name", "Node", "GPU", "Started on", "Duration", "Status")
		for j := range joblist[i] {
			job := joblist[i][j]
			Node := fmt.Sprintf("%02d", job.Node)
			GPU := fmt.Sprintf("%d", job.Gpu)
			StartTime := job.StartTime.Format("01-Sep-2006 15:04:05")
			Duration := job.Runtime.String()

			var Status string
			switch job.Status {
			case -1:
				Status = "FAILED"
			case 0:
				Status = "FINISHED"
			case 1:
				Status = "RUNNING"
			case 2:
				Status = "PENDING"
			}
			body += fmt.Sprintf("<tr class='%s'>", strings.ToLower(Status))
			if job.Status != 2 {
				body += fmt.Sprintf("<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</tr>\n",
					"          ", job.Name, Node, GPU, StartTime, Duration, Status)
			} else {
				const None = "---"
				body += fmt.Sprintf("<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n</tr>\n",
					"          ", job.Name, None, None, None, None, Status)
			}
		}
		body += "</table>\n"
	}
	body += "</table>"
	body += fmt.Sprint(htmlFooter)
	output := ([]byte)(body)
	filename := *out + "/" + OUTNAME
	_, errA := os.Stat(filename)
	if !os.IsNotExist(errA) {
		os.Remove(filename)
	}
	file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0666)
	defer file.Close()
	if err != nil {
		return
	}
	_, err1 := file.Write(output)
	if err1 != nil {
		return
	}
}

func main() {

	flag.Parse()
	if len(*out) == 0 {
		*out, _ = os.Getwd()
	}
	_, err := os.Stat(*home)
	if os.IsNotExist(err) {
		fmt.Printf("%s does not exist\nAborting\n", *home)
	}
	//fmt.Printf("Looking into %s directory...\n", *home)
	GetListOfUsers()
	filelist = make([][]string, len(users))
	joblist = make([][]Job, len(users))

	for i := range users {
		GetDaemonFiles(i)
		ProcessJobFromLog(i)
	}
	//DumpQue()
	DumpQueHtml()
}
