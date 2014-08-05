package main

import (
	"fmt"
	"os/exec"
	"strings"
)

type GPU struct {
	info string
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
			gpus = append(gpus, GPU{info})
		}
	}
	return gpus
}
