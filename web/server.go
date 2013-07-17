package web

import (
	"log"
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

	log.Println("web interface unavailable")

	//	s := gui.NewServer(templText)
	//
	//	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	//	go func() {
	//		util.LogErr(s.ListenAndServe(port))
	//	}()
	//	runtime.Gosched()
}
