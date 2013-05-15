package web

// Handlers for web script input "script/"

import (
	"code.google.com/p/mx3/engine"
	"net/http"
)

func scriptHandler(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/script/"):]
	engine.Inject <- func() { engine.Exec(cmd) } // TODO: catch
}
