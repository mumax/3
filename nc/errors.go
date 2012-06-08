package nc

// Error handling utilities.

import (
	"fmt"
	"log"
	"runtime"
)

// Panics on the message.
func Panic(msg ...interface{}) {
	panic(fmt.Sprint(msg...))
}

// Checks for an IO error.
func CheckIO(err error) {
	if err != nil {
		panic(err)
	}
}

// Logs the error of non-nil.
func CheckLog(err error) {
	if err != nil {
		log.Println("[error  ]", err)
	}
}

// Logs.
func Log(msg ...interface{}) {
	log.Println(msg...)
}

// If test == false, panic with the file and
// line number of this function's caller. 
func Assert(test bool) {
	if !test {
		msg := "assertion failed"
		_, file, line, ok := runtime.Caller(1)
		if ok {
			msg += ": " + file + ":" + fmt.Sprint(line)
		}
		panic(msg)
	}
}
