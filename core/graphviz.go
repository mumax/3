package core

// This file implements graphviz output.

import (
	"fmt"
	"io"
	"os/exec"
)

var Graph *Graphviz

func InitGraphviz(fname string) {
	if Graph != nil{
		Fatal(fmt.Errorf("already saving pipeline graph"))
	}
	Log("saving pipeline graph to", fname)
	Graph = &Graphviz{OpenFile(fname), fname}
	Graph.Println("digraph dot{")
	Graph.Println("rankdir=LR")
	AtExit(func(){Graph.Close()})
}

type Graphviz struct {
	out   io.WriteCloser
	fname string
}

func (g *Graphviz) Println(msg ...interface{}) {
	if g.out == nil{return}
	fmt.Fprintln(g.out, msg...)
}

func (g *Graphviz) Close() {
	g.Println("}")
	g.out.Close()
	LogErr(exec.Command("dot", "-O", "-Tpdf", g.fname).Run())
}
func (g *Graphviz) Connect(dst string, src string, label string, thickness int) {
	if dst == "" || src == "" {
		Panic("connect", dst, src, label, thickness)
	}
	g.Println(src, "->", dst, "[label=", escape(label), `penwidth=`, thickness, `];`)
}


func (g *Graphviz) Add(box string) {
	g.Println(box, `[label="`+box+`"shape="rect"];`)
}

// replaces characters that graphviz cannot handle as labels.
func escape(in string) string {
	return `"` + in + `"`
}
