package util

import (
	"fmt"
	"os"
	"time"
)

var lastdash = time.Now()

func Dashf(format string, v ...interface{}) {
	if time.Since(lastdash) > 100*time.Millisecond {
		lastdash = time.Now()
		fmt.Fprintf(os.Stderr, format, v...)
		fmt.Fprint(os.Stderr, "\u000D")
	}
}

func DashExit() {
	fmt.Println(os.Stderr)
}
