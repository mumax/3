//go:build !windows

package mag

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// availableMemory returns the available physical memory in bytes (MemAvailable from /proc/meminfo),
// 0 if unknown. Used to size CalcDemagKernelFast's image cache.
func availableMemory() int64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var kb int64
		if strings.HasPrefix(sc.Text(), "MemAvailable:") {
			fmt.Sscanf(sc.Text(), "MemAvailable: %d kB", &kb)
			return kb * 1024
		}
	}
	return 0
}
