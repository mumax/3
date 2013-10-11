package engine

import (
	"github.com/mumax/3/cuda"
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
	rescaleBuf  *data.Slice // GPU
	imgBuf      *data.Slice // CPU
	img         = new(image.NRGBA)
	renderScale = 1
	renderLayer = 0
	renderComp  string
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
		log.Println(err)
		http.Error(w, err, http.StatusNotFound)
		return
	} else {
		render(q, comp)
		jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	}
}

func render(quant Slicer, comp string) {

	//	defer func() { // TODO: remove
	//		err := recover()
	//		if err != nil {
	//			log.Println("render:", err)
	//		}
	//	}()

	size := quant.Mesh().Size()

	// don't slice out of bounds
	if renderLayer >= size[0] {
		renderLayer = size[0] - 1
		// TODO: update slider
	}
	if renderLayer < 0 {
		renderLayer = 0
	}
	if quant.NComp() == 1 {
		renderComp = ""
	}

	// scale the size
	for i := range size {
		size[i] /= renderScale
		if size[i] == 0 {
			size[i] = 1
		}
	}
	size[0] = 1 // selects one layer

	// make sure buffers are there
	if imgBuf.Mesh().Size() != size {
		mesh := data.NewMesh(size[0], size[1], size[2], 1, 1, 1)
		imgBuf = data.NewSlice(3, mesh) // always 3-comp, may be re-used
	}

	// rescale and download
	InjectAndWait(func() {

		//defer func() { // TODO: remove
		//	err := recover()
		//	if err != nil {
		//		log.Println("render inject:", err)
		//	}
		//}()

		buf, r := quant.Slice()
		if r {
			defer cuda.Recycle(buf)
		}
		if !buf.GPUAccess() {
			log.Println("no GPU access for", quant.Name)
			imgBuf = Download(quant) // fallback (no zoom)
			return
		}

		// make sure buffers are there (in CUDA context)
		if rescaleBuf.Mesh().Size() != size {
			mesh := data.NewMesh(size[0], size[1], size[2], 1, 1, 1)
			rescaleBuf.Free()
			rescaleBuf = cuda.NewSlice(1, mesh)
		}

		for c := 0; c < quant.NComp(); c++ {
			cuda.Resize(rescaleBuf, buf.Comp(c), renderLayer)
			data.Copy(imgBuf.Comp(c), rescaleBuf)
		}
	})

	d := imgBuf
	if comp != "" && d.NComp() > 1 {
		c := compstr[comp]
		d = d.Comp(c)
	}
	draw.On(img, d, "auto", "auto")
}
