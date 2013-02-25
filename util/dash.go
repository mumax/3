package util

import (
	"fmt"
	"time"
)

var lastdash = time.Now()

func Dash(step, undone int, t, dt, err float64) {
	if *Flag_silent {
		return
	}
	if time.Since(lastdash) > 100*time.Millisecond {
		lastdash = time.Now()
		fmt.Printf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e\u000D", step, undone, float32(t), float32(dt), float32(err))
	}
}

func DashExit() {
	if *Flag_silent {
		return
	}
	fmt.Println()
}
