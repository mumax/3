package main

import (
	"bufio"
	"flag"
	"io"
	"nimble-cube/core"
	"os"
)

func main() {
	flag.Parse()
	if flag.NArg() == 0 {
		read(os.Stdin)
	}
	for _, arg := range flag.Args() {
		in, err := os.Open(arg)
		core.Fatal(err)
		read(bufio.NewReader(in))
		in.Close()
	}
}

func read(in io.Reader) {
}
