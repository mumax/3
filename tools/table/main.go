package main

// Author: Arne Vansteenkiste

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"flag"
	"fmt"
	"io"
	"os"
)

var (
	flag_noheader = flag.Bool("noheader", false, "Omit header")
)

func main() {
	flag.Parse()

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
	r := dump.NewTableReader(in)
	if !*flag_noheader {
		fmt.Print("# ")
		for i := range r.Tags {
			fmt.Print(r.Tags[i], " (", r.Units[i], ")\t")
		}
		fmt.Println()
	}
	err := r.ReadLine()
	for err == nil {
		for _, x := range r.Data {
			fmt.Print(x, "\t")
		}
		fmt.Println()
		err = r.ReadLine()
	}
}
