package engine

import (
	"fmt"
)

var hist string

func Log(msg ...interface{}) {
	m := fmt.Sprintln(msg...)
	if len(m) > 1000 {
		m = m[:1000-3] + "...\n"
	}
	hist += m
	fmt.Print(m)
}
