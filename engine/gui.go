package engine

import (
	"code.google.com/p/mx3/gui"
	"code.google.com/p/mx3/util"
	"log"
	"net/http"
	"runtime"
)

var GUI *gui.Doc

// Start web gui on given port, does not block.
func GoServe(port string) {

	GUI = gui.NewDoc("/", templText)
	//http.HandleFunc("/render/", render)

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	go func() {
		util.LogErr(http.ListenAndServe(port, nil))
	}()
	runtime.Gosched()
}
