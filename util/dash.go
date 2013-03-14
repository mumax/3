package util

import (
	"fmt"
	"time"
)

var lastdash = time.Now()

func Dashf(format string, v ...interface{}) {
	if time.Since(lastdash) > 100*time.Millisecond {
		lastdash = time.Now()
		fmt.Printf(format, v...)
		fmt.Print("\u000D")
	}
}

func DashExit() {
	fmt.Println()
}
