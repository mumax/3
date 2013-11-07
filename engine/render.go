package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/util"
	"image"
	"image/jpeg"
	"net/http"
	"strings"
)

var compstr = map[string]int{"x": 0, "y": 1, "z": 2}

// this is GUI state
var (
	rescaleBuf  *data.Slice // GPU
	imgBuf      *data.Slice // CPU
	img         = new(image.NRGBA)
	renderScale = 1    // zoom for render
	renderLayer = 0    // layer to render
	renderComp  string // component to render
)

// Render image of quantity.
// Accepts url: /render/name and /render/name/component
func serveRender(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len("/render/"):]
	words := strings.Split(url, "/")
	quant := words[0]
	comp := renderComp
	if len(words) > 1 {
		comp = words[1]
	}
	if quant == "" {
		quant = renderQ
	}
	q, ok := quants[quant]
	if !ok {
		err := "render: unknown quantity: " + url
		util.Log(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		render(q, comp)
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}

func render(quant Slicer, comp string) {
	size := quant.Mesh().Size()

	// don't slice out of bounds
	if renderLayer >= size[Z] {
		renderLayer = size[Z] - 1
	}
	if renderLayer < 0 {
		renderLayer = 0
	}
	if quant.NComp() == 1 {
		comp = ""
	}

	// scale the size
	for i := range size {
		size[i] /= renderScale
		if size[i] == 0 {
			size[i] = 1
		}
	}
	size[Z] = 1 // selects one layer

	// make sure buffers are there
	if imgBuf.Mesh().Size() != size {
		mesh := data.NewMesh(size[X], size[Y], size[Z], 1, 1, 1)
		imgBuf = data.NewSlice(3, mesh) // always 3-comp, may be re-used
	}

	// rescale and download
	InjectAndWait(func() {

		buf, r := quant.Slice()
		if r {
			defer cuda.Recycle(buf)
		}
		if !buf.GPUAccess() {
			util.Log("no GPU access for", quant.Name)
			imgBuf = Download(quant) // fallback (no zoom)
			return
		}

		// make sure buffers are there (in CUDA context)
		if rescaleBuf.Mesh().Size() != size {
			mesh := data.NewMesh(size[X], size[Y], size[Z], 1, 1, 1)
			rescaleBuf.Free()
			rescaleBuf = cuda.NewSlice(1, mesh)
		}

		for c := 0; c < quant.NComp(); c++ {
			cuda.Resize(rescaleBuf, buf.Comp(c), renderLayer)
			data.Copy(imgBuf.Comp(c), rescaleBuf)
		}
	})

	// imgBuf always has 3 components, we may need just one...
	d := imgBuf
	if comp != "" && quant.NComp() > 1 { // ... if one has been selected by gui
		d = d.Comp(compstr[comp])
	}
	if quant.NComp() == 1 { // ...or if the original data only had one (!)
		d = d.Comp(0)
	}
	draw.On(img, d, "auto", "auto")
}
