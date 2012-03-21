package mx

import (
	"fmt"
	"runtime"
)

// Panics if test is false
func Assert(test bool) {
	if !test {
		_, file, line, _ := runtime.Caller(1)
		panic(fmt.Sprint("assertion failed:", file, ":", line))
	}
}
