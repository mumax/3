package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
	"runtime"
)

var webPort string

func GoServe(port string) {

	if webPort != "" {
		return // already serving
	}
	webPort = port

	http.HandleFunc("/cmd/top", Command("top", "-b", "-n", "1"))
	http.HandleFunc("/cmd/uname", Command("uname", "-a"))

	http.HandleFunc("/render/", render)

	http.HandleFunc("/ctl/", control)

	http.HandleFunc("/", gui)

	log.Print("serving GUI on http://localhost", port, "\n")
	go func() {
		cuda.LockThread()
		util.FatalErr(http.ListenAndServe(port, nil))
	}()
	runtime.Gosched()
}
