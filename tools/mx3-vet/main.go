/*
 This tool checks input scripts for common mistakes. Useful, e.g., before submitting them to a cluster.
 Usage:
 	mx3-vet files
*/
package main

import (
	"code.google.com/p/mx3/engine"
	"flag"
	"fmt"
)

func main() {
	flag.Parse()
	for _, f := range flag.Args() {
		fmt.Println(f)
		engine.Vet(f)
	}
}
