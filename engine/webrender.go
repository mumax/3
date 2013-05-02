package engine

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/draw"
	"image/jpeg"
	"log"
	"net/http"
)

// render image of quantity
func render(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len("/render/"):]
	h, ok := quants[url]
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

type downloader interface {
	Download() *data.Slice
}

var quants = make(map[string]downloader)
