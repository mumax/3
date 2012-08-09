package main

import (
	"flag"
	"io"
	"nimble-cube/core"
	"os"
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

}
