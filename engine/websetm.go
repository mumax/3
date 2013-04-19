package engine

// Handlers for web control "ctl/"

import (
	"log"
	"net/http"
)

func setm(w http.ResponseWriter, r *http.Request) {
	arg := r.FormValue("value")
	log.Println("setm", arg)
	http.Redirect(w, r, "/", http.StatusFound)
}
