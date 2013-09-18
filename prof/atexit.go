package prof

import "log"

// Functions to be called at program exit
var atexit []func()

// Add a function to be executed at program exit.
func AtExit(cleanup func()) {
	atexit = append(atexit, cleanup)
}

// Runs all functions stacked by AtExit().
func Cleanup() {
	if len(atexit) != 0 {
		log.Println("stopping profiler")
	}
	for _, f := range atexit {
		f()
	}
}
