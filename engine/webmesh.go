package engine

// Handlers for mesh setting through web interface

import (
	"log"
	"net/http"
)

func setmesh(w http.ResponseWriter, r *http.Request) {
	pause()
	ui.Lock()
	defer ui.Unlock()

	log.Println("setmesh")

	http.Redirect(w, r, "/", http.StatusFound)
}
