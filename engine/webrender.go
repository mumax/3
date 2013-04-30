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

// map of names to Handle does not work because Handles change on the fly
// *Handle does not work because we loose interfaceness.
func quant(name string) (h buffered_iface, ok bool) {
	switch name {
	default:
		return nil, false
	case "m":
		return M, true
	case "torque":
		return Torque, true
	}
	return nil, false // rm for go 1.1
}

// Output handle that also support manual single-shot saving.
// TODO: replace by smallest struct/iface that captures Get()
type buffered_iface interface {
	Download() *data.Slice // CPU-accessible slice
}
