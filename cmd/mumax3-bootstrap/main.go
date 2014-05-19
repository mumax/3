// Bootstrap looks for a mumax+cuda combination that works on your system.
package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"strings"
)

const LD_LIBRARY_PATH = "LD_LIBRARY_PATH"
const PATH = "PATH"

var env []string

func main() {

	sep := string(os.PathListSeparator)
	bin := execDir()
	//fmt.Println("bin:", bin)

	env = os.Environ()
	for i := range env {
		if strings.HasPrefix(env[i], LD_LIBRARY_PATH+"=") {
			env[i] += sep + bin
			check(os.Setenv(LD_LIBRARY_PATH, env[i][len(LD_LIBRARY_PATH):]))
		}
		if strings.HasPrefix(env[i], PATH+"=") {
			env[i] += sep + bin
			//fmt.Println(env[i])
		}
	}

	cmds := []string{"mumax3-cuda5.0", "mumax3-cuda5.5", "mumax3-cuda6.0"}
	mumax := "mumax3-cuda5.5" // default
	for _, cmd := range cmds {
		cmd := bin + "/" + cmd
		err := run(cmd, []string{"-test"}, false)
		if err == nil {
			mumax = cmd
			break
		}
	}

	fmt.Println("running", mumax)
	err := run(mumax, os.Args[1:], true)
	if err != nil {
		fatal(err)
	}
}

func run(command string, args []string, pipe bool) error {
	// prepare command
	cmd := exec.Command(command, args...)
	cmd.Env = env

	done := make(chan int)
	var stdout, stderr io.ReadCloser
	if pipe {
		stdout, _ = cmd.StdoutPipe()
		stderr, _ = cmd.StderrPipe()
	}

	// prepare output
	//outfname := "test.out" // TODO: proper name
	outfile := ioutil.Discard
	//f, erro := os.OpenFile(outfname, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	//if erro == nil {
	//	defer f.Close()
	//	outfile = f
	//} else {
	//	log.Println(erro)
	//}
	if pipe {
		multiOut := io.MultiWriter(os.Stdout, outfile)
		multiErr := io.MultiWriter(os.Stderr, outfile)
		go func() { io.Copy(multiOut, stdout); done <- 1 }()
		go func() { io.Copy(multiErr, stderr); done <- 1 }()
	}

	// run & flush output
	err := cmd.Run()
	if pipe {
		<-done
		<-done
	}

	return err
}

// try really hard to get the executable's directory
func execDir() string {
	dir, err := os.Readlink("/proc/self/exe")
	if err == nil {
		return path.Dir(dir)
	}
	log.Println(err)
	dir, err = exec.LookPath("mumax3")
	if err == nil {
		return path.Dir(dir)
	}
	log.Println(err)
	return "."
}

func check(e error) {
	if e != nil {
		fatal(e)
	}
}

func fatal(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
	os.Exit(1)
}
