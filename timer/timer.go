package timer

import (
	"fmt"
	"io"
	"sort"
	"time"
)

var (
	clocks     map[string]*clock
	firstStart time.Time
	Timeout    time.Duration
)

func Start(key string) {
	if clocks == nil {
		clocks = make(map[string]*clock)
		firstStart = time.Now()
	}

	if c, ok := clocks[key]; ok {
		c.Start()
	} else {
		clocks[key] = new(clock)
		// do not start, first run = warmup time
	}
}

func Stop(key string) {
	clocks[key].Stop()
}

type clock struct {
	total       time.Duration
	started     time.Time
	invocations int
}

func (c *clock) Start() {
	c.started = time.Now()
	c.invocations++
}

func (c *clock) Stop() {
	if (c.started == time.Time{}) {
		return // not started
	}
	d := time.Since(c.started)
	c.total += d
	c.started = time.Time{}
	if Timeout != 0 && d > Timeout {
		panic("launch timeout: " + d.String())
	}
}

// entry for sorted output by Print()
type entry struct {
	name        string
	total       time.Duration
	invocations int
	pct         float32
}

func (e *entry) String() string {
	perOp := time.Duration(int64(e.total) / int64(e.invocations))
	return fmt.Sprint(pad(e.name), pad(fmt.Sprint(e.invocations, "x")), perOp, "/op\t", e.pct, " %\t", e.total, " total")
}

func pad(s string) string {
	if len(s) >= 20 {
		return s
	}
	return s + "                    "[:20-len(s)]
}

type entries []entry

func (l entries) Len() int           { return len(l) }
func (l entries) Less(i, j int) bool { return l[i].total > l[j].total }
func (l entries) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }

func Print(out io.Writer) {
	if clocks == nil {
		return
	}
	wallTime := time.Since(firstStart)
	lines := make(entries, 0, len(clocks))
	var accounted time.Duration
	for k, v := range clocks {
		pct := 100 * float32(int64(v.total)) / float32(int64(wallTime))
		lines = append(lines, entry{k, v.total, v.invocations, pct})
		accounted += v.total
	}

	unaccounted := wallTime - accounted
	pct := 100 * float32(int64(unaccounted)) / float32(int64(wallTime))
	lines = append(lines, entry{"NOT TIMED", unaccounted, 1, pct})

	sort.Sort(lines)

	for _, l := range lines {
		fmt.Fprintln(out, &l)
	}
}
