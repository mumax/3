package main

import (
	"flag"
	"nimble-cube/dump"
)

func main() {
	flag.Parse()
	frames := dump.ReadAllFiles(flag.Args(), dump.CRC_ENABLED)
	for f := range frames {
		f.Print()
	}
}
