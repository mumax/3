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

type getter interface {
	Download() *data.Slice
}

var quants = map[string]getter{}

// map of names to Handle does not work because Handles change on the fly
// *Handle does not work because we loose interfaceness.
func quant(name string) (h getter, ok bool) {
	h, ok = quants[name]
	return
}
