package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"strings"
)

var env []string

const LD_LIBRARY_PATH = "LD_LIBRARY_PATH"

func main() {
	bin := execDir()

	env = os.Environ()
	for i := range env {
		if strings.HasPrefix(env[i], LD_LIBRARY_PATH+"=") {
			env[i] += ":" + bin
			fmt.Println(env[i])
		}
	}

	cmds := []string{"mumax3-cuda5.5", "mumax3-cuda5.0", "mumax3-cuda4.2.9"}
	mumax := ""
	for _, cmd := range cmds {
		cmd := bin + "/" + cmd
		err := run(cmd, []string{"-test"})
		if err == nil {
			mumax = cmd
			break
		}
	}

	if mumax == "" {
		fatal("no matching mumax/cuda combination found in", cmds)
	}
	fmt.Println(mumax, os.Args[1:])
	err := run(mumax, os.Args[1:])
	if err != nil {
		fatal(err)
	}
}

func run(command string, args []string) error {
	cmd := exec.Command(command, args...)
	cmd.Env = env
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	done := make(chan int)
	go func() { io.Copy(os.Stdout, stdout); done <- 1 }()
	go func() { io.Copy(os.Stderr, stderr); done <- 1 }()
	err := cmd.Run()
	<-done
	<-done
	return err
}

func fatal(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
	os.Exit(1)
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
