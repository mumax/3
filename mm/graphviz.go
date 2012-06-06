package mm

import (
	"fmt"
	"io"
	"log"
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
	if err != nil {
		log.Println(err) //far from fatal
		return
	}
	dot.Println("digraph dot{")
	dot.Println("rankdir=LR")
}

func (dot *graphvizwriter) Println(msg ...interface{}) {
	dot.Init()
	fmt.Fprintln(dot.out, msg...)
}

func (dot *graphvizwriter) Connect(dst string, src string, label string, thickness int) {
	dot.Init()
	dot.Println(src, `[shape="rect"];`)
	dot.Println(dst, `[shape="rect"];`)
	dot.Println(src, "->", dst, "[label=", label, `penwidth=`, thickness, `];`)
}

func (dot *graphvizwriter) Close() {
	dot.Println("}")
	dot.out.Close()
	err := exec.Command("dot", "-O", "-Tpdf", dot.fname).Run()
	if err != nil {
		log.Println(err)
	}
}
func boxname(value interface{}) string {
	typ := fmt.Sprintf("%T", value)
	return typ[strings.Index(typ, ".")+1:]
}
