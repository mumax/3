package main

import (
	"flag"
	"io"
	"nimble-cube/core"
	"nimble-cube/dump"
	"os"
)

var (
	flag_crc  = flag.Bool("crc", true, "Generate/check CRC checksums")
	flag_show = flag.Bool("show", false, "Human-readible output to stdout")
)

func main() {
	flag.Parse()
	core.LOG = false

	if flag.NArg() == 0 {
		read(os.Stdin, "")
	}
	for _, arg := range flag.Args() {
		f, err := os.Open(arg)
		core.Fatal(err)
		read(f, arg)
		f.Close()
	}
}

func read(in io.Reader, name string) {
	r := dump.NewReader(in, *flag_crc)
	err := r.Read()
	for err != io.EOF {
		core.Fatal(err)
		process(r, name)
		err = r.Read()
	}
}

func process(r *dump.Reader, name string) {
	haveOutput := false

	if !haveOutput || *flag_show {
		r.Fprint(os.Stdout)
		haveOutput = true
	}
}
