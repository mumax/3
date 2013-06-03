package engine

// File: initialization of command line flags.
// Author: Arne Vansteenkiste

import (
	"log"
	"runtime"
)

const VERSION = "mx3.0.8 α "

var UNAME = VERSION + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + "(" + runtime.Compiler + ")"

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {
	log.Println("shutting down")
	drainOutput()
	Table.flush()
}
