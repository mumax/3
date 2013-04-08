package web

import (
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
)

func Serve(port string) {

	http.HandleFunc("/cmd/top", Command("top", "-b", "-n", "1"))
	http.HandleFunc("/cmd/uname", Command("uname", "-a"))

	http.HandleFunc("/render/", render)

	log.Print("serving http://localhost:", port, "\n")
	util.FatalErr(http.ListenAndServe(port, nil))
}
