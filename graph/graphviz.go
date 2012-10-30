package core

// This file implements graphviz output.

import (
	"fmt"
	"io"
	"nimble-cube/core"
	"os/exec"
)

var global *writer

func Init(fname string) {
	if global != nil {
		core.Fatal(fmt.Errorf("already saving pipeline graph"))
	}
	core.Log("saving pipeline graph to", fname)
	global = &writer{core.OpenFile(fname), fname}
	global.Println("digraph dot{")
	global.Println("rankdir=LR")
	core.AtExit(func() { global.Close() })
}

type writer struct {
	out   io.WriteCloser
	fname string
}

func (g *writer) Println(msg ...interface{}) {
	if g.out == nil {
		return
	}
	fmt.Fprintln(g.out, msg...)
}

func (g *writer) Close() {
	g.Println("}")
	g.out.Close()
	core.LogErr(exec.Command("dot", "-O", "-Tpdf", g.fname).Run())
}
func (g *writer) Connect(dst string, src string, label string, thickness int) {
	if dst == "" || src == "" {
		core.Panic("connect", dst, src, label, thickness)
	}
	g.Println(src, "->", dst, "[label=", escape(label), `penwidth=`, thickness, `];`)
}

func (g *writer) Add(box string) {
	g.Println(box, `[label="`+box+`"shape="rect"];`)
}

// replaces characters that graphviz cannot handle as labels.
func escape(in string) string {
	return `"` + in + `"`
}
