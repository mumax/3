package nimble

// This file implements graphviz output.

import (
	"code.google.com/p/nimble-cube/core"
	"fmt"
	"io"
	"os/exec"
)

var global *writer

func InitGraph(fname string) {
	if global != nil {
		core.Fatalf("already saving pipeline graph")
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
	if g == nil {
		return
	}
	fmt.Fprintln(g.out, msg...)
}

func (g *writer) Close() {
	g.Println("}")
	g.out.Close()
	dot := exec.Command("dot", "-O", "-Tpdf", g.fname)
	out, err := dot.CombinedOutput()
	if err != nil {
		core.Log("dot:", string(out))
		core.Log("dot:", err)
	}
}

func Connect(src, dst string) {
	global.Connect(src, dst)
}

func (g *writer) Connect(src, dst string) {
	//	if dst == "" || src == "" {
	//		Panic("connect", dst, src, label, thickness)
	//	}
	g.Println(src, "->", dst) //, "[label=", escape(label), `penwidth=`, thickness, `];`)
}

func AddQuant(tag string) {
	global.AddQuant(tag)
}

func (g *writer) AddQuant(tag string) {
	g.Println(tag, `[label="`+tag+`"shape="oval"];`)
}

func AddRoutine(tag string) {
	global.AddRoutine(tag)
}

func (g *writer) AddRoutine(tag string) {
	g.Println(tag, `[label="`+tag+`"shape="rect"];`)
}

// replaces characters that graphviz cannot handle as labels.
func escape(in string) string {
	return `"` + in + `"`
}
