package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/draw"
	"image/jpeg"
	"log"
	"net/http"
	"strings"
)

var compstr = map[string]int{"x": 2, "y": 1, "z": 0} // also swaps XYZ user space

// Render image of quantity.
// Accepts url: /render/name and /render/name/component
func serveRender(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len("/render/"):]
	words := strings.Split(url, "/")
	quant := words[0]
	comp := ""
	if len(words) > 1 {
		comp = words[1]
	}
	if quant == "" {
		quant = renderQ
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
		InjectAndWait(func() { d = download(h) })
		if comp != "" && d.NComp() > 1 {
			c := compstr[comp]
			d = d.Comp(c)
		}
		img := draw.Image(d, "auto", "auto")
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}

// Download a quantity to host,
// or just return its data when already on host.
func download(q Getter) *data.Slice {
	buf, recycle := q.Get()
	if recycle {
		defer cuda.RecycleBuffer(buf)
	}
	if buf.CPUAccess() {
		return buf
	} else {
		return buf.HostCopy()
	}
}
