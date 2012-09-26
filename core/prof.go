package core

// Profiler

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"text/tabwriter"
	"time"
)

var (
	tags      = make(map[string]bool)
	timeline  = make([]*stamp, 0, 1000)
	profstate sync.Mutex
	profstart time.Time
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
	profstart = time.Now()
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
	out := tabwriter.NewWriter(out_, 8, 1, 1, ' ', 0)
	var keys []string
	for k, _ := range tags {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for i, s := range timeline {
		del := "|"
		if i%4 == 0 {
			del = "- - -"
		}
		tags[s.tag] = (s.delta >= 0)
		fmt.Fprintf(out, "%15v ", s.Time.Sub(profstart))
		for _, k := range keys {
			if tags[k] == true {
				fmt.Fprint(out, "\t", k)
			} else {
				fmt.Fprint(out, "\t"+del)
			}
		}
		fmt.Fprintln(out)
		out.Flush()
	}
}
