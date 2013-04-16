package engine

// Handlers for web set parameters "set/"

import (
	"log"
	"net/http"
	"strconv"
)

func set(w http.ResponseWriter, r *http.Request) {
	//cmd := r.URL.Path[len("/set/"):]

	key := "msat"
	arg := r.FormValue(key)
	v, err := strconv.ParseFloat(arg, 64)
	if err != nil {
		http.Error(w, key+": "+err.Error(), 400)
		return
	}
	Msat = Const(v)
	log.Println("msat:", v)

	key = "aex"
	arg = r.FormValue(key)
	v, err = strconv.ParseFloat(arg, 64)
	if err != nil {
		http.Error(w, key+": "+err.Error(), 400)
		return
	}
	Aex = Const(v)
	log.Println("aex:", v)

	http.Redirect(w, r, "/", http.StatusFound)
}
