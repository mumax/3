package web

import (
	"code.google.com/p/mx3/draw"
	"code.google.com/p/mx3/engine"
	"image/jpeg"
	"net/http"
	"strings"
)

func render(w http.ResponseWriter, r *http.Request) {
	url := strings.ToLower(r.URL.Path[len("/render/"):])
	h, ok := getQuant(url)
	if !ok {
		http.Error(w, "render: unknown quantity: "+url, http.StatusNotFound)
		return
	} else {
		img := draw.Image(h.Download(), "auto", "auto") // TODO: not very concurrent
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}

func getQuant(name string) (h engine.Buffered, ok bool) {
	switch name {
	case "m":
		return engine.M, true
	}
	return nil, false
}
