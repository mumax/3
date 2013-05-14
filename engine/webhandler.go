package engine

import (
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
	"runtime"
)

var webPort string

// start web gui server
func ServeHTTP() {
	goServe(*flag_port)
}

// Start web gui on given port, does not block.
func goServe(port string) {

	// already serving or don't want to serve
	if webPort != "" || port == "" {
		return
	}
	webPort = port

	http.HandleFunc("/cmd/top", command("top", "-b", "-n", "1"))
	http.HandleFunc("/cmd/uname", command("uname", "-a"))

	http.HandleFunc("/render/", render)
	http.HandleFunc("/dash/", dash)

	http.HandleFunc("/ctl/", control)
	http.HandleFunc("/setparam/", setparam)
	http.HandleFunc("/setmesh/", setmesh)
	http.HandleFunc("/running/", isrunning)
	http.HandleFunc("/setm/", setm)

	http.HandleFunc("/", gui)
	http.HandleFunc("/script/", scriptHandler)

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	go func() {
		util.LogErr(http.ListenAndServe(port, nil)) // should not be fatal, but then we should not open browser.
	}()
	runtime.Gosched()
}
