package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/draw"
	"image/jpeg"
	"log"
	"net/http"
	"strings"
)

// render image of quantity
func render(w http.ResponseWriter, r *http.Request) {
	url := strings.ToLower(r.URL.Path[len("/render/"):])
	log.Println("render", url)
	h, ok := Quant(url)
	if !ok {
		err := "render: unknown quantity: " + url
		log.Println(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		cuda.LockThread()                               // TODO: for bootstrapping only, use dedicated thread
		img := draw.Image(h.Download(), "auto", "auto") // TODO: not very concurrent
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}
