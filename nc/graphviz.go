package nc

// This file implements graphviz output.

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

type graphvizwriter struct {
	fname string
	out   io.WriteCloser
}

func (dot *graphvizwriter) Init(fname string) {
	if dot.out != nil {
		return // already inited.
	}
	dot.fname = fname
	var err error
	dot.out, err = os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	CheckLog(err)
	dot.Println("digraph dot{")
	dot.Println("rankdir=LR")
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

func (dot *graphvizwriter) AddBox(name string) {
	dot.Println(name, `[shape="rect"];`)
}

func (dot *graphvizwriter) Close() {
	dot.Println("}")
	dot.out.Close()
	err := exec.Command("dot", "-O", "-Tpdf", dot.fname).Run()
	CheckLog(err)
}

// replaces characters that graphviz cannot handle as labels.
func escape(in string) (out string) {
	return `"` + in + `"`
	return
}
