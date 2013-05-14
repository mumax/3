package engine

// Handlers for web script input "script/"

import (
	"log"
	"net/http"
)

func scriptHandler(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/script/"):]
	log.Println("web script:", cmd)
}
