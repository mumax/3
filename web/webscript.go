package engine

// Handlers for web script input "script/"

import (
	"fmt"
	"log"
	"net/http"
)

func scriptHandler(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/script/"):]
	log.Println("web script:", cmd)
	code, err := parser.ParseLine(cmd)
	if err != nil {
		fmt.Fprintln(w, err)
	}
	inject <- func() { code.Eval() } // TODO: catch
}

// inject to pause simulation.
func pauseFn() { pause = true }

func init() {
	parser.AddFunc("pause", pauseFn)
}
