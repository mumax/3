//go:build !windows

// This file must provide the getCPUInfo() function for non-Windows systems.

package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strings"
)

func getCPUInfo() string {
	// Check the runtime operating system
	switch runtime.GOOS {
	case "linux":
		return getLinuxCPUInfo()
	// Add more cases for other operating systems if needed
	default:
		return fmt.Sprintf("CPU info: Unknown OS: %s", runtime.GOOS)
	}
}

func getLinuxCPUInfo() string {
	file, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return fmt.Sprintf("CPU info: Unknown, Error: %s", err.Error())
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var cpuDetails []string
	var cpuModel, cpuCores, cpuMHz string
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, ":")
		if len(fields) != 2 {
			continue
		}
		key := strings.TrimSpace(fields[0])
		value := strings.TrimSpace(fields[1])
		switch key {
		case "model name":
			cpuModel = value
		case "cpu cores":
			cpuCores = value
		case "cpu MHz":
			cpuMHz = value
		}
	}
	if cpuModel != "" && cpuCores != "" && cpuMHz != "" {
		cpuDetails = append(cpuDetails, fmt.Sprintf("CPU info: %s, Cores: %s, MHz: %s", cpuModel, cpuCores, cpuMHz))
	}

	return strings.Join(cpuDetails, "; ")
}
