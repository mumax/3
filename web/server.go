package web

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
	"runtime"
)

func GoServe(port string) {

	http.HandleFunc("/cmd/top", Command("top", "-b", "-n", "1"))
	http.HandleFunc("/cmd/uname", Command("uname", "-a"))

	http.HandleFunc("/render/", render)

	http.HandleFunc("/", gui)

	log.Print("serving http://localhost", port, "\n")
	go func() {
		cuda.LockThread()
		util.FatalErr(http.ListenAndServe(port, nil))
	}()
	runtime.Gosched()
}
