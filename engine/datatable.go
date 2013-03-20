package engine

import (
	"bufio"
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"fmt"
	"os"
)

var tabout *bufio.Writer

func savetable() {
	if Table.needSave() {
		initTable()
		writeTableLine()
		Table.count++
	}
}

func writeTableLine() {
	ncell := float64(m.Mesh().NCell())
	mx := float64(cuda.Sum(mx)) / ncell
	my := float64(cuda.Sum(my)) / ncell
	mz := float64(cuda.Sum(mz)) / ncell
	fmt.Fprintln(tabout, Time, mx, my, mz)
	tabout.Flush()
}

// make sure tabout is open.
func initTable() {
	if tabout == nil {
		f, err := os.OpenFile(OD+"datatable.txt", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		util.FatalErr(err)
		tabout = bufio.NewWriter(f)
		fmt.Fprintln(tabout, "# t(s) mx my mz")
		tabout.Flush()
	}
}
