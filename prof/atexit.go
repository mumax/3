package prof

// File: manages functions to be called at program exit
// Author: Arne Vansteenkiste

import "log"

var atexit []func()

// Add a function to be executed at program exit.
func AtExit(cleanup func()) {
	atexit = append(atexit, cleanup)
}

// Runs all functions stacked by AtExit().
func Cleanup() {
	if len(atexit) != 0 {
		log.Println("cleanup")
	}
	for _, f := range atexit {
		f()
	}
}
