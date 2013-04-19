package engine

// Handlers for web control "ctl/"

import (
	"log"
	"net/http"
)

func setm(w http.ResponseWriter, r *http.Request) {
	arg := r.FormValue("value")
	log.Println("setm", arg)

	err := setMFile(arg)
	if err != nil {
		http.Error(w, "set magnetization: "+err.Error(), 400)
		return
	}

	http.Redirect(w, r, "/", http.StatusFound)
}
