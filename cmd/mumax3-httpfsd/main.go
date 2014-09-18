package main

import (
	"flag"
	"github.com/mumax/3/httpfs"
	"log"
	"net/http"
)

var (
	flag_v = flag.Bool("v", false, "enable logging")
	flag_p = flag.String("p", ":35371", "http listen address")
)

func main() {
	flag.Parse()
	httpfs.Logging = *flag_v
	httpfs.Handle()
	log.Println("serving", *flag_p)
	err := http.ListenAndServe(*flag_p, nil)
	if err != nil {
		log.Fatal(err)
	}
}
