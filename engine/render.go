package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	//"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
	"image"
	"image/jpeg"
	"net/http"
	"sync"
)

var renderer = render{img: new(image.NRGBA)}

type render struct {
	saveCountLock sync.Mutex
	saveCount     map[Quantity]int
	prevSaveCount int // previous max slider value of time

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

	comp := GUI.StringValue("renderComp")
	quant := GUI.StringValue("renderQuant")
	q, ok := GUI.Quants[quant]
	if !ok {
		err := "render: unknown quantity: " + quant
		util.Log(err)
		http.Error(w, err, http.StatusNotFound)
		return
	}
	ren.render(q, comp)
	jpeg.Encode(w, ren.img, &jpeg.Options{Quality: 100})
}

// rescale and download quantity, save in rescaleBuf
func (ren *render) download(quant Quantity, comp string) {
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
		GUI.Set("renderLayerLabel", fmt.Sprint(renderLayer, "/", Mesh().Size()[Z]))
		if quant.NComp() == 1 {
			comp = ""
			GUI.Set("renderComp", "")
		}
		// scale the size
		renderScale := maxScale - GUI.IntValue("renderScale")
		GUI.Set("renderScaleLabel", fmt.Sprint("1/", renderScale))
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
}

//func (ren *render) registerSaveCount(q Quantity, autonum int) {
//	ren.saveCountLock.Lock()
//	defer ren.saveCountLock.Unlock()
//	if ren.saveCount == nil {
//		ren.saveCount = make(map[Quantity]int)
//	}
//	ren.saveCount[q] = autonum
//}
//
//func (ren *render) getSaveCount(q Quantity) int {
//	ren.saveCountLock.Lock()
//	defer ren.saveCountLock.Unlock()
//	if ren.saveCount == nil {
//		return 0
//	}
//	return ren.saveCount[q]
//}

// TODO: have browser cache the images
func (ren *render) getQuant(quant Quantity, comp string) {
	//rTime := GUI.IntValue("renderTime")
	//saveCount := ren.getSaveCount(quant)
	//GUI.Attr("renderTime", "max", saveCount)

	//// if we were "live" (extreme right), keep right
	//if rTime == ren.prevSaveCount {
	//	rTime = saveCount
	//	GUI.Set("renderTime", rTime)
	//}
	//ren.prevSaveCount = saveCount

	//if rTime == saveCount { // live view
	ren.download(quant, comp)
	GUI.Set("renderTimeLabel", Time)
	//} else {
	//	defer func() {
	//		if err := recover(); err != nil {
	//			LogOutput(err)
	//		}
	//	}()
	//	slice, info, err := oommf.ReadFile(autoFname(quant.Name(), rTime))
	//	if err != nil {
	//		LogOutput(err)
	//		return
	//	}
	//	println("read ovf")
	//	GUI.Set("renderTimeLabel", info.Time)
	//	ren.imgBuf = slice
	//}
}

func (ren *render) render(quant Quantity, comp string) {
	ren.getQuant(quant, comp)
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
