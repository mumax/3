package engine

// Handlers for web set parameters "set/"

import (
	"fmt"
	"log"
	"net/http"
	"strconv"
)

func set(w http.ResponseWriter, r *http.Request) {

	ui.Lock()
	defer ui.Unlock()

	for name, inf := range meta_inf {
		vals := make([]float64, inf.NComp())
		for c := range inf.Comp() {
			str := r.FormValue(fmt.Sprint(name, c))
			v, err := strconv.ParseFloat(str, 64)
			if err != nil {
				http.Error(w, "set "+name+": "+err.Error(), 400)
				return
			}
			vals[c] = v
		}
		have := inf.Get()
		if !eq(have, vals) {
			log.Println("set", name, vals)
			inf.Set(vals)
		}
	}

	http.Redirect(w, r, "/", http.StatusFound)
}

func eq(a, b []float64) bool {
	for i, v := range a {
		if b[i] != v {
			return false
		}
	}
	return true
}
