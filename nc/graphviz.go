package nc

// This file implements the plumber's graphviz output

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

var dot graphvizwriter

type graphvizwriter struct {
	fname string
	out   io.WriteCloser
}

func (dot *graphvizwriter) Init() {
	if dot.out != nil {
		return // already inited.
	}

	dot.fname = "plumber.dot"
	var err error

	dot.out, err = os.OpenFile(dot.fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	CheckLog(err)
	dot.Println("digraph dot{")
	dot.Println("rankdir=LR")
}

func (dot *graphvizwriter) Println(msg ...interface{}) {
	dot.Init()
	fmt.Fprintln(dot.out, msg...)
}

func (dot *graphvizwriter) Connect(dst string, src string, label string, thickness int) {
	dot.Init()
	if dst == "" || src == "" {
		Panic("connect", dst, src, label, thickness)
	}
	dot.Println(src, "->", dst, "[label=", label, `penwidth=`, thickness, `];`)
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

func boxname(value interface{}) string {
	typ := fmt.Sprintf("%T", value)
	clean := typ[strings.Index(typ, ".")+1:] // strip "*mm."
	if strings.HasSuffix(clean, "Box") {
		clean = clean[:len(clean)-len("Box")]
	}
	return clean
}
