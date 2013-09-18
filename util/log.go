package util

// Logging and error reporting utility functions

import (
	"log"
	//"fmt"
	//"path"
	//"runtime"
)

// If err != nil, trigger log.Fatal(msg, err)
func FatalErr(err interface{}, msg ...interface{}) {
	if err != nil {
		log.Fatal(append(msg, err)...)
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

// Debug message, includes file + line.
//func Debug(msg ...interface{}) {
//	_, file, line, ok := runtime.Caller(1)
//	file = path.Base(file)
//	caller := ""
//	if ok {
//		caller = fmt.Sprint(file, ":", line, ":")
//	}
//	log.Println(append([]interface{}{caller}, msg...)...)
//}

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
