/*
httpfs server, useful for debugging mumax3-server.

# Usage

Start mumax3-httpfsd in a certain working directory.

	$ ls
	file.mx3

	$ mumax3-server -l :35362

Then you can remotely run mumax3 input files:

	$ cd elsewhere
	$ mumax3 http://localhost:35362/file.mx3
*/
package main

import (
	"flag"
	"log"
	"net/http"
	_ "net/http/pprof"

	"github.com/mumax/3/httpfs"
)

var (
	flag_addr = flag.String("l", ":35360", "Listen and serve at this network address")
	flag_log  = flag.Bool("log", false, "log debug output")
)

func main() {
	flag.Parse()
	log.Println("serving at", *flag_addr)
	httpfs.Logging = *flag_log
	httpfs.RegisterHandlers()
	err := http.ListenAndServe(*flag_addr, nil)
	if err != nil {
		log.Fatal(err)
	}
}
