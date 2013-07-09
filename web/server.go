package web

import (
	"code.google.com/p/mx3/gui"
	"code.google.com/p/mx3/util"
	"log"
	"runtime"
)

// Start web gui on given port, does not block.
func GoServe(port string) {

	//	http.HandleFunc("/render/", render)
	//	http.HandleFunc("/dash/", dashHandler)
	//
	//	http.HandleFunc("/running/", runningHandler)
	//
	//	http.HandleFunc("/", gui)
	//	http.HandleFunc("/script/", scriptHandler)

	s := gui.NewServer(templText)

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	go func() {
		util.LogErr(s.ListenAndServe(port))
	}()
	runtime.Gosched()
}
