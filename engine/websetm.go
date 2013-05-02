package engine

import (
	"log"
	"net/http"
)

// Handler that sets magnetization from file
func setm(w http.ResponseWriter, r *http.Request) {
	arg := r.FormValue("value")
	log.Println("setm", arg)

	err := M.setFile(arg)
	if err != nil {
		http.Error(w, "set magnetization: "+err.Error(), 400)
		return
	}

	http.Redirect(w, r, "/", http.StatusFound)
}
