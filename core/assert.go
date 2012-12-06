package core

import (
	"fmt"
	"runtime"
)

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
