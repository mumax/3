// +build ignore

/*
Toy interpreter executes files or stdin.
*/
package main

import (
	. "."
	"flag"
	"fmt"
	"io/ioutil"
	"os"
)

var debug = flag.Bool("g", false, "print debug output")

func main() {
	flag.Parse()
	w := NewWorld()
	Debug = *debug
	if flag.NArg() > 0 {
		for _, arg := range flag.Args() {
			src, err := ioutil.ReadFile(arg)
			check(err)
			check(w.Exec(string(src)))
		}
	} else {

		fmt.Fprintln(os.Stderr, "Reading from stdin")
		src, err := ioutil.ReadAll(os.Stdin)
		check(err)
		check(w.Exec(string(src)))
	}
}

func check(e error) {
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
