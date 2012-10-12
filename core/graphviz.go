package core

// This file implements graphviz output.

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

var graphout *graphvizwriter

type graphvizwriter struct {
	fname string
	out   io.WriteCloser
}

func newGraphvizWriter(fname string) *graphvizwriter {
	dot := new(graphvizwriter)
	dot.fname = fname
	var err error
	dot.out, err = os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	LogErr(err)
	dot.Println("digraph dot{")
	dot.Println("rankdir=LR")
	return dot
}

func (dot *graphvizwriter) Println(msg ...interface{}) {
	fmt.Fprintln(dot.out, msg...)
}

func (dot *graphvizwriter) Connect(dst string, src string, label string, thickness int) {
	if dst == "" || src == "" {
		Panic("connect", dst, src, label, thickness)
	}
	dot.Println(src, "->", dst, "[label=", escape(label), `penwidth=`, thickness, `];`)
}

func (dot *graphvizwriter) AddBox(box string) {
	dot.Println(box, `[label="`+box+`"shape="rect"];`)
}

func (dot *graphvizwriter) Close() {
	dot.Println("}")
	dot.out.Close()
	err := exec.Command("dot", "-O", "-Tpdf", dot.fname).Run()
	LogErr(err)
}

// replaces characters that graphviz cannot handle as labels.
func escape(in string) (out string) {
	return `"` + in + `"`
	return
}
