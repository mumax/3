package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/mumax/3/engine"
	"github.com/mumax/3/util"
)

// check all input files for errors, don't run.
func vet() {
	status := 0
	for _, f := range flag.Args() {
		src, ioerr := ioutil.ReadFile(f)
		util.FatalErr(ioerr)
		engine.World.EnterScope() // avoid name collisions between separate files
		_, err := engine.World.Compile(string(src))
		engine.World.ExitScope()
		if err != nil {
			fmt.Println(f, ":", err)
			status = 1
		} else {
			fmt.Println(f, ":", "OK")
		}
	}
	os.Exit(status)
}
