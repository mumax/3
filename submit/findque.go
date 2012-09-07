package main

import (
	"os"
	"path"
)

const MagicFile = ".here_be_your_mumax_que__d3990c0c658a293ac278bb591f15d031"

// find the que directory based on hidden file.
func findque(wd string) string {
	if len(wd) == 0 {
		return ""
	}
	if wd[0] != '/' {
		return ""
	}
	for len(wd) > 1 { // handles "/" and "."
		wd = path.Dir(wd) // start with parent, should not submit from actual que dir
		if _, err := os.Stat(wd + "/que/" + MagicFile); err == nil {
			return wd + "/que"
		}
	}
	return ""
}
