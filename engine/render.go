package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"image"
	"image/jpeg"
	"math"
	"net/http"
	"sync"
)

//var renderer = render{img: new(image.RGBA)}

type render struct {
	mutex        sync.Mutex
	quant        Quantity
	comp         string
	layer, scale int
	saveCount    int         // previous max slider value of time
	rescaleBuf   *data.Slice // GPU
	imgBuf       *data.Slice // CPU
	img_         *image.RGBA
}

const (
	maxScale   = 32  // maximum zoom-out setting
	maxImgSize = 512 // maximum render image size
)

// Render image of quantity.
func (g *guistate) ServeRender(w http.ResponseWriter, r *http.Request) {
	g.render.mutex.Lock()
	defer g.render.mutex.Unlock()

	g.render.render()
	jpeg.Encode(w, g.render.img_, &jpeg.Options{Quality: 100})
}

// rescale and download quantity, save in rescaleBuf
func (ren *render) download() {
	InjectAndWait(func() {
		if ren.quant == nil { // not yet set, default = m
			ren.quant = &M
		}
		quant := ren.quant
		size := quant.Mesh().Size()

		// don't slice out of bounds
		renderLayer := ren.layer
		if renderLayer >= size[Z] {
			renderLayer = size[Z] - 1
		}
		if renderLayer < 0 {
			renderLayer = 0
		}

		// scaling sanity check
		if ren.scale < 1 {
			ren.scale = 1
		}
		if ren.scale > maxScale {
			ren.scale = maxScale
		}
		// Don't render too large images or we choke
		for size[X]/ren.scale > maxImgSize {
			ren.scale++
		}
		for size[Y]/ren.scale > maxImgSize {
			ren.scale++
		}

		for i := range size {
			size[i] /= ren.scale
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
}

var arrowSize = 16

func (ren *render) render() {
	ren.download()
	// imgBuf always has 3 components, we may need just one...
	d := ren.imgBuf
	comp := ren.comp
	quant := ren.quant
	if comp == "" {
		normalize(d)
	}
	if comp != "" && quant.NComp() > 1 { // ... if one has been selected by gui
		d = d.Comp(compstr[comp])
	}
	if quant.NComp() == 1 { // ...or if the original data only had one (!)
		d = d.Comp(0)
	}
	if ren.img_ == nil {
		ren.img_ = new(image.RGBA)
	}
	draw.On(ren.img_, d, "auto", "auto", arrowSize)
}

var compstr = map[string]int{"x": 0, "y": 1, "z": 2}

func normalize(f *data.Slice) {
	a := f.Vectors()
	maxnorm := 0.
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {

				x, y, z := a[0][i][j][k], a[1][i][j][k], a[2][i][j][k]
				norm := math.Sqrt(float64(x*x + y*y + z*z))
				if norm > maxnorm {
					maxnorm = norm
				}

			}
		}
	}
	factor := float32(1 / maxnorm)

	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				a[0][i][j][k] *= factor
				a[1][i][j][k] *= factor
				a[2][i][j][k] *= factor

			}
		}
	}
}
