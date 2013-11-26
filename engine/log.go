package engine

import (
	"fmt"
)

var hist string

func Log(msg ...interface{}) {
	m := fmt.Sprintln(msg...)
	m = m[:len(m)-1] // strip newline
	if len(m) > 1000 {
		m = m[:1000-3] + "..."
	}
	if hist != "" { // prepend newline
		hist += "\n"
	}
	hist += m
	fmt.Println(m)
	if GUI.Page != nil {
		GUI.Set("console", hist)
	}
}
