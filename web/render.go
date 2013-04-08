package web

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/draw"
	"code.google.com/p/mx3/engine"
	"image/jpeg"
	"net/http"
	"strings"
)

// render image of quantity
func render(w http.ResponseWriter, r *http.Request) {
	url := strings.ToLower(r.URL.Path[len("/render/"):])
	h, ok := engine.Quant(url)
	if !ok {
		http.Error(w, "render: unknown quantity: "+url, http.StatusNotFound)
		return
	} else {
		cuda.LockThread()                               // TODO: for bootstrapping only, use dedicated thread
		img := draw.Image(h.Download(), "auto", "auto") // TODO: not very concurrent
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}
