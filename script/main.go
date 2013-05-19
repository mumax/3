// +build ignore

package main

import (
	. "."
	"flag"
	"fmt"
)

func main() {
	flag.Parse()
	w := NewWorld()
	for _, arg := range flag.Args() {
		fmt.Println(w.MustEval(arg))
	}
}
