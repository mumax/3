package engine

// File: initialization of command line flags.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/mx3/prof"
	"flag"
	"log"
	"runtime"
)

const VERSION = "mx3.0.7 α "

var UNAME = VERSION + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + "(" + runtime.Compiler + ")"

// Initializes the simulation engine.
func Init() {
	flag.Parse()

}

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {

	log.Println("shutting down")
	drainOutput()
	Table.flush()
	prof.Cleanup()
}
