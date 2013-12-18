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
	"sync"
)

var renderer = render{img: new(image.NRGBA)}

type render struct {
	mutex      sync.Mutex
	rescaleBuf *data.Slice // GPU
	imgBuf     *data.Slice // CPU
	img        *image.NRGBA
}

const maxScale = 32

// Render image of quantity.
// Accepts url: /render/name and /render/name/component
func (ren *render) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ren.mutex.Lock()
	defer ren.mutex.Unlock()

	url := r.URL.Path[len("/render/"):]
	words := strings.Split(url, "/")
	quant := words[0]
	comp := GUI.StringValue("renderComp")
	if len(words) > 1 {
		comp = words[1]
	}
	if quant == "" {
		quant = GUI.StringValue("renderQuant")
	}
	q, ok := GUI.Quants[quant]
	if !ok {
		err := "render: unknown quantity: " + url
		util.Log(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		ren.render(q, comp)
		jpeg.Encode(w, ren.img, &jpeg.Options{Quality: 100})
	}
}

func (ren *render) render(quant Slicer, comp string) {
	// rescale and download
	InjectAndWait(func() {
		size := quant.Mesh().Size()

		// don't slice out of bounds
		renderLayer := GUI.IntValue("renderLayer")
		if renderLayer >= size[Z] {
			renderLayer = size[Z] - 1
			GUI.Set("renderLayer", renderLayer)
		}
		if renderLayer < 0 {
			renderLayer = 0
			GUI.Set("renderLayer", renderLayer)
		}
		if quant.NComp() == 1 {
			comp = ""
			GUI.Set("renderComp", "")
		}

		// scale the size
		renderScale := maxScale - GUI.IntValue("renderScale")
		for i := range size {
			size[i] /= renderScale
			if size[i] == 0 {
				size[i] = 1
			}
		}
		size[Z] = 1 // selects one layer

		// make sure buffers are there
		if ren.imgBuf.Size() != size {
			ren.imgBuf = data.NewSlice(3, size) // always 3-comp, may be re-used
		}

		buf, r := quant.Slice()
		if r {
			defer cuda.Recycle(buf)
		}
		if !buf.GPUAccess() {
			ren.imgBuf = Download(quant) // fallback (no zoom)
			return
		}

		// make sure buffers are there (in CUDA context)
		if ren.rescaleBuf.Size() != size {
			ren.rescaleBuf.Free()
			ren.rescaleBuf = cuda.NewSlice(1, size)
		}

		for c := 0; c < quant.NComp(); c++ {
			cuda.Resize(ren.rescaleBuf, buf.Comp(c), renderLayer)
			data.Copy(ren.imgBuf.Comp(c), ren.rescaleBuf)
		}
	})

	// imgBuf always has 3 components, we may need just one...
	d := ren.imgBuf
	if comp != "" && quant.NComp() > 1 { // ... if one has been selected by gui
		d = d.Comp(compstr[comp])
	}
	if quant.NComp() == 1 { // ...or if the original data only had one (!)
		d = d.Comp(0)
	}
	draw.On(ren.img, d, "auto", "auto")
}

var compstr = map[string]int{"x": 0, "y": 1, "z": 2}
