//go:build darwin

package main

import (
	"strings"
	"testing"
)

func TestDarwinSystemInfo(t *testing.T) {
	if got := getOSInfo(); !strings.HasPrefix(got, "macOS ") {
		t.Fatalf("getOSInfo() = %q, want a macOS version", got)
	}

	cpuInfo := getCPUInfo()
	if !strings.HasPrefix(cpuInfo, "CPU info: ") ||
		strings.Contains(cpuInfo, "Unknown") {
		t.Fatalf("getCPUInfo() = %q, want a detected CPU", cpuInfo)
	}
}
