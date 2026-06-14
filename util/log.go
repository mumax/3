package util

// Logging and error reporting utility functions

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
)

func Fatal(msg ...interface{}) {
	log.Fatal(msg...)
}

func Fatalf(format string, msg ...interface{}) {
	log.Fatalf(format, msg...)
}

// If err != nil, trigger log.Fatal(msg, err)
func FatalErr(err interface{}) {
	_, file, line, _ := runtime.Caller(1)
	if err != nil {
		log.Fatal(file, ":", line, err)
	}
}

// Panics if err is not nil. Signals a bug.
func PanicErr(err error) {
	if err != nil {
		log.Panic(err)
	}
}

// Logs the error of non-nil, plus message
func LogErr(err error, msg ...interface{}) {
	if err != nil {
		log.Println(append(msg, err)...)
	}
}

func Log(msg ...interface{}) {
	log.Println(msg...)
}

// Panics with "illegal argument" if test is false.
func Argument(test bool) {
	if !test {
		log.Panic("illegal argument")
	}
}

// Panics with msg if test is false
func AssertMsg(test bool, msg interface{}) {
	if !test {
		log.Panic(msg)
	}
}

// Panics with "assertion failed" if test is false.
func Assert(test bool) {
	if !test {
		log.Panic("assertion failed")
	}
}

// Hack to avoid cyclic dependency on engine.
var (
	progress_ func(int, int, string) = PrintProgress
	progLock  sync.Mutex
)

// Set progress bar to progress/total and display msg
// if GUI is up and running.
func Progress(progress, total int, msg string) {
	progLock.Lock()
	defer progLock.Unlock()
	if progress_ != nil {
		progress_(progress, total, msg)
	}
}

var (
	lastPct   = -1      // last progress percentage shown
	lastProgT time.Time // last time we showed progress percentage
)

func PrintProgress(prog, total int, msg string) {
	pct := (prog * 100) / total
	if pct != lastPct { // only print percentage if changed
		if (time.Since(lastProgT) > time.Second) || pct == 100 || prog == 0 { // only print percentage once/second unless finished
			fmt.Println("//", msg, pct, "%")
			lastPct = pct
			lastProgT = time.Now()
		}
	}
}

// Sets the function to be used internally by Progress.
// Avoids cyclic dependency on engine.
func SetProgress(f func(int, int, string)) {
	progLock.Lock()
	defer progLock.Unlock()
	progress_ = f
}
