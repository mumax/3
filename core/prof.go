package core

// Profiler

import (
	"fmt"
	"io"
	"sync"
	"text/tabwriter"
	"time"
)

var (
	tags      = make(map[string]bool)
	timeline  = make([]*stamp, 0, 1000)
	profstate sync.Mutex
)

type stamp struct {
	tag   string
	delta int
	time.Time
}

func profRegister(tag string) {
	profstate.Lock()
	defer profstate.Unlock()
	// make sure tag is not yet in use
	if _, ok := tags[tag]; ok {
		Panic("prof: tag", tag, "already in use")
	}
	tags[tag] = false
}

func profWriteNext(tag string, delta int) {
	profstate.Lock()
	timeline = append(timeline, &stamp{tag, delta, time.Now()})
	profstate.Unlock()
}

func profWriteDone(tag string) {
	profstate.Lock()
	timeline = append(timeline, &stamp{tag, -1, time.Now()})
	profstate.Unlock()
}

func ProfDump(out_ io.Writer) {
	out := tabwriter.NewWriter(out_, 0, 8, 1, ' ', 0)
	for _, s := range timeline {
		tags[s.tag] = (s.delta >= 0)
		fmt.Fprint(out, s.Time)
		for tag, on := range tags {
			if on {
				fmt.Fprint(out, tag, "\n")
			} else {
				fmt.Fprint(out, " \t")
			}
		}
		fmt.Fprintln(out)
	}
}
