package engine

import (
	"code.google.com/p/mx3/data"
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
	h, ok := quant(url)
	if !ok {
		err := "render: unknown quantity: " + url
		log.Println(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		var d *data.Slice
		injectAndWait(func() { d = h.Download() })
		img := draw.Image(d, "auto", "auto")
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}
