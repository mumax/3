package engine

import (
	"bufio"
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"fmt"
	"os"
)

type dataTable struct {
	*bufio.Writer
	autosave
}

func (t *dataTable) send(M *data.Synced, good bool) {
	if good && t.needSave() {
		t.init()
		m := M.Read()
		ncell := float64(m.Mesh().NCell())
		mx := float64(cuda.Sum(m.Comp(2))) / ncell
		my := float64(cuda.Sum(m.Comp(1))) / ncell
		mz := float64(cuda.Sum(m.Comp(0))) / ncell
		M.ReadDone()
		fmt.Fprintln(t, Time, mx, my, mz)
		t.saved()
	}
}

func newTable(name string) *dataTable {
	t := new(dataTable)
	t.name = name
	return t
}

// make sure tabout is open.
func (t *dataTable) init() {
	if t.Writer == nil {
		f, err := os.OpenFile(OD+t.name+".txt", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		util.FatalErr(err)
		t.Writer = bufio.NewWriter(f)
		fmt.Fprintln(t, "# t(s) mx my mz")
		t.Flush()
	}
}

func (t *dataTable) flush() {
	if t.Writer != nil {
		t.Flush()
	}
}
