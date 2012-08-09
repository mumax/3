package main

import (
	"flag"
	"io"
	"nimble-cube/core"
	"nimble-cube/dump"
	"os"
)

var (
	flag_crc = flag.Bool("crc", true, "Generate/check CRC64 checksums.")
)

func main() {
	flag.Parse()
	core.LOG = false

	if flag.NArg() == 0 {
		process(os.Stdin)
	} else {
		for _, arg := range flag.Args() {
			f, err := os.Open(arg)
			core.Fatal(err)
			process(f)
			f.Close()
		}
	}
}

func process(in io.Reader) {
	r := dump.NewReader(in, *flag_crc)
	err := r.Read()
	for err != io.EOF {
		core.Fatal(err)
		r.Fprint(os.Stdout)
		err = r.Read()
	}
}
