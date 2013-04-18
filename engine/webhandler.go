package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
	"runtime"
)

var webPort string

// Start web gui on given port, does not block.
func GoServe(port string) {

	if webPort != "" {
		return // already serving
	}
	webPort = port

	http.HandleFunc("/cmd/top", Command("top", "-b", "-n", "1"))
	http.HandleFunc("/cmd/uname", Command("uname", "-a"))

	http.HandleFunc("/render/", render)
	http.HandleFunc("/dash/", dash)

	http.HandleFunc("/ctl/", control)
	http.HandleFunc("/setparam/", setparam)
	http.HandleFunc("/setmesh/", setmesh)
	http.HandleFunc("/running/", isrunning)

	http.HandleFunc("/", gui)

	log.Print("serving GUI on http://localhost", port, "\n")
	go func() {
		cuda.LockThread()
		util.FatalErr(http.ListenAndServe(port, nil)) // should not be fatal, but then we should not open browser.
	}()
	runtime.Gosched()
}
