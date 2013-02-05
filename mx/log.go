package mx

// Logging and error reporting utility functions

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
		fmt.Fprintln(os.Stderr, append(msg, err)...)
		Cleanup()
		os.Exit(1)
	}
}

func FatalExit(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
	Cleanup()
	os.Exit(1)
}

func Fatalf(format string, args ...interface{}) {
	FatalErr(fmt.Errorf(format, args...))
}

// Panics on the message.
func Panic(msg ...interface{}) {
	panic(fmt.Errorf(fmt.Sprint(msg...)))
}

// Panics on the message.
func Panicf(format string, args ...interface{}) {
	panic(fmt.Sprintf(format, args...))
}

// Panics if err is not nil
func PanicErr(err error) {
	if err != nil {
		panic(err)
	}
}

// Logs the error of non-nil.
func LogErr(err error) {
	if err != nil {
		Log(err)
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

//func Check(test bool, msg string) {
//	if !test {
//		Fatal(fmt.Errorf(msg))
//	}
//}

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
