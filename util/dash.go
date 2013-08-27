package util

import (
	"fmt"
	"os"
	"time"
)

var (
	lastdash    = time.Now()
	dashVisible bool
	DashEnable  bool
)

func Dashf(format string, v ...interface{}) {
	if DashEnable && time.Since(lastdash) > 100*time.Millisecond {
		lastdash = time.Now()
		fmt.Fprintf(os.Stderr, format, v...)
		fmt.Fprint(os.Stderr, "\u000D")
		dashVisible = true
	}
}

func DashExit() {
	if dashVisible {
		fmt.Fprintln(os.Stderr)
		dashVisible = false
	}
}
