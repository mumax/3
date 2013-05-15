package web

import (
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
	"runtime"
)

// Start web gui on given port, does not block.
func GoServe(port string) {

	http.HandleFunc("/render/", render)
	http.HandleFunc("/dash/", dashHandler)

	http.HandleFunc("/running/", runningHandler)

	http.HandleFunc("/", gui)
	http.HandleFunc("/script/", scriptHandler)

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	go func() {
		util.LogErr(http.ListenAndServe(port, nil)) // should not be fatal, but then we should not open browser.
	}()
	runtime.Gosched()
}
