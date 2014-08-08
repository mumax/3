package main

import (
	"fmt"
	"log"
	"os/exec"
	"strings"
)

type GPU struct {
	Info string
}

const MAXGPU = 16

func DetectGPUs() []GPU {
	var gpus []GPU
	for i := 0; i < MAXGPU; i++ {
		gpuflag := fmt.Sprint("-gpu=", i)
		out, err := exec.Command(*flag_mumax, "-test", gpuflag).Output()
		if err == nil {
			info := string(out)
			if strings.HasSuffix(info, "\n") {
				info = info[:len(info)-1]
			}
			log.Println("gpu", i, ":", info)
			gpus = append(gpus, GPU{info})
		}
	}
	return gpus
}

func DetectMumax() string {
	out, err := exec.Command(*flag_mumax, "-test", "-v").Output()
	if err == nil {
		info := string(out)
		split := strings.SplitN(info, "\n", 2)
		version := split[0]
		log.Println("have", version)
		return version
	}
	return ""
}
