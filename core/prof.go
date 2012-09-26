package core

// Profiler

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"text/tabwriter"
	"time"
	//"github.com/ajstarks/svgo"

)

var (
	tags      = make(map[string]bool)
	timeline  = make([]*stamp, 0, 1000)
	profstate sync.Mutex
	profstart time.Time
	keys      []string
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
	profstate.Lock()
	defer profstate.Unlock()
	out := tabwriter.NewWriter(out_, 8, 1, 1, ' ', 0)
	profUpdateKeys()
	for _, s := range timeline {
		//	// delimiter
		//	del := "|"
		//	if i%4 == 0 {
		//		del = "- - -"
		//	}

		// enable/disable "running" status for this tag
		tags[s.tag] = (s.delta >= 0)

		// print status of all tags
		profPrintTags(out, s)
	}
}

func profUpdateKeys() {
	keys = keys[:0]
	for k, _ := range tags {
		keys = append(keys, k)
	}
	sort.Strings(keys)
}

var del = "|"

func profPrintTags(out *tabwriter.Writer, s *stamp) {
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

//func ProfDumpSVG(fname string){
//	profstate.Lock()
//	defer profstate.Unlock()
//	out := OpenFile(fname)
//	defer out.Close()
//
//    width := 500
//    height := 500
//    canvas := svg.New(out)
//
//	profUpdateKeys()
//	for _, s := range timeline {
//		// enable/disable "running" status for this tag
//		tags[s.tag] = (s.delta >= 0)
//
//		for _, k := range keys {
//			if tags[k] == true {
//				fmt.Fprint(out, "\t", k)
//			} else {
//				fmt.Fprint(out, "\t"+del)
//			}
//	}
//	}
//    canvas.Start(width, height)
//    canvas.Circle(width/2, height/2, 100)
//    canvas.Text(width/2, height/2, "Hello, SVG", "text-anchor:middle;font-size:30px;fill:black")
//    canvas.End()
//}
