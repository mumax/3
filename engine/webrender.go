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
	url := r.URL.Path[len("/render/"):]
	words := strings.Split(url, "/")
	quant := words[0]
	comp := ""
	if len(words) > 1 {
		comp = words[1]
	}
	h, ok := quants[quant]
	if !ok {
		err := "render: unknown quantity: " + url
		log.Println(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		var d *data.Slice
		// TODO: could download only needed component
		injectAndWait(func() { d = h.Download() })
		if comp != "" && d.NComp() > 1 {
			c := compstr[comp]
			d = d.Comp(c)
		}
		img := draw.Image(d, "auto", "auto")
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}

var compstr = map[string]int{"x": 2, "y": 1, "z": 0} // also swaps XYZ user space

type downloader interface {
	Download() *data.Slice
}

var quants = make(map[string]downloader)
