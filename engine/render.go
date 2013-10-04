package engine

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"image"
	"image/jpeg"
	"log"
	"net/http"
	"strings"
)

var compstr = map[string]int{"x": 2, "y": 1, "z": 0} // also swaps XYZ user space

var (
	rescaleBuf *data.Slice // GPU
	imgBuf     *data.Slice // CPU
	img        = new(image.NRGBA)
)

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
	q, ok := quants[quant]
	if !ok {
		err := "render: unknown quantity: " + url
		log.Println(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		render(q, comp)
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}

func render(quant Slicer, comp string) {
	var d *data.Slice

	InjectAndWait(func() { d = Download(quant) })

	if comp != "" && d.NComp() > 1 {
		c := compstr[comp]
		d = d.Comp(c)
	}
	draw.On(img, d, "auto", "auto")
}
