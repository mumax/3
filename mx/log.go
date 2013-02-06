package mx

// File: logging and error reporting utility functions
// Author: Arne Vansteenkiste

import (
	"fmt"
	"log"
	"os"
	"path"
	"runtime"
)

var (
	LOG   = true
	DEBUG = true
)

// called by init()
func initLog() {
	LOG = !*Flag_silent
	DEBUG = *Flag_debug
	log.SetPrefix(" ·")
	log.SetFlags(0)
}

// If err != nil, print the message and error, run Cleanup and exit.
// E.g.:
// 	f, err := os.Open(file)
// 	FatalErr(err, "open", file)
// May output:
// 	open /some/file file does not exist
func FatalErr(err interface{}, msg ...interface{}) {
	if err != nil {
		FatalExit(append(msg, err)...)
	}
}

// Print message to stderr, run Cleanup and exit unsuccessfully.
func FatalExit(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
	cleanup()
	os.Exit(1)
}

// Like FatalExit, but with Printf formatting.
func Fatalf(format string, args ...interface{}) {
	FatalErr(fmt.Errorf(format, args...))
}

// Panics on the message. Signals a bug.
func Panic(msg ...interface{}) {
	panic(fmt.Errorf(fmt.Sprint(msg...)))
}

// Panics on the message, with Printf formatting.
// Signals a bug.
func Panicf(format string, args ...interface{}) {
	panic(fmt.Sprintf(format, args...))
}

// Panics if err is not nil. Signals a bug.
func PanicErr(err error) {
	if err != nil {
		panic(err)
	}
}

// Logs the error of non-nil, plus message
func LogErr(err error, msg ...interface{}) {
	if err != nil {
		Log(append(msg, err)...)
	}
}

// Log message.
func Log(msg ...interface{}) {
	if LOG {
		log.Println(msg...)
	}
}

// Log message.
func Logf(format string, args ...interface{}) {
	if LOG {
		log.Printf(format, args...)
	}
}

// Debug message.
func Debug(msg ...interface{}) {
	if DEBUG {
		_, file, line, ok := runtime.Caller(1)
		file = path.Base(file)
		caller := ""
		if ok {
			caller = fmt.Sprint(file, ":", line, ":")
		}
		Log(append([]interface{}{caller}, msg...)...)
	}
}
