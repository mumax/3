package web

import (
	"code.google.com/p/mx3/draw"
	"code.google.com/p/mx3/engine"
	"net/http"
	"strings"
)

var handles = map[string]engine.Buffered{"m": engine.M, "torque": engine.Torque}

func render(w http.ResponseWriter, r *http.Request) {
	url := strings.ToLower(r.URL.Path[len("/render/"):])
	h, ok := handles[url]
	if !ok {
		http.Error(w, "render: unknown quantity: "+url, http.StatusNotFound)
		return
	}
}
