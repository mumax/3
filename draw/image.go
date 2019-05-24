package draw

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"image"
	"image/color"
	"log"
	"strconv"
)

// Renders an image of slice. fmin, fmax = "auto" or a number to set the min/max color scale.
func Image(f *data.Slice, fmin, fmax string, arrowSize int, colormap ...ColorMapSpec) *image.RGBA {
	img := new(image.RGBA)
	On(img, f, fmin, fmax, arrowSize, colormap...)
	return img
}

// Render on existing image buffer. Resize it if needed
func On(img *image.RGBA, f *data.Slice, fmin, fmax string, arrowSize int, colormap ...ColorMapSpec) {
	dim := f.NComp()
	switch dim {
	default:
		log.Fatalf("unsupported number of components: %v", dim)
	case 3:
		if colormap == nil {
			drawVectors(img, f.Vectors(), arrowSize)
			break
		}
		if colormap[0].Ccomp >= 0 {
			ff := f.Comp(colormap[0].Ccomp)
			min, max := parseMinMax(ff, fmin, fmax)
			drawFloats(img, ff.Scalars(), min, max, colormap[0].Cmap...)
			if arrowSize > 0 {
				drawArrows(img, f.Vectors(), arrowSize)
			}
		} else {
			drawVectors(img, f.Vectors(), arrowSize)
		}

	case 1:
		min, max := parseMinMax(f, fmin, fmax)
		if colormap == nil {
			drawFloats(img, f.Scalars(), min, max)
		} else {
			drawFloats(img, f.Scalars(), min, max, colormap[0].Cmap...)
		}
	}
}

func parseMinMax(f *data.Slice, fmin, fmax string) (min, max float32) {
	min, max = extrema(f.Host()[0])
	if fmin != "auto" {
		m, err := strconv.ParseFloat(fmin, 32)
		if err != nil {
			util.Fatal("draw: scale:", err)
		}
		min = float32(m)
	}
	if fmax != "auto" {
		m, err := strconv.ParseFloat(fmax, 32)
		if err != nil {
			util.Fatal("draw: scale:", err)
		}
		max = float32(m)
	}
	if min == max {
		min -= 1
		max += 1 // make it gray instead of black
	}
	return
}

// Draws rank 4 tensor (3D vector field) as image
// averages data over X (usually thickness of thin film)
func drawVectors(img *image.RGBA, arr [3][][][]float32, arrowSize int) {
	w, h := len(arr[X][0][0]), len(arr[X][0])
	d := len(arr[X])
	norm := float32(d)
	*img = *recycle(img, w, h)
	for iy := 0; iy < h; iy++ {
		for ix := 0; ix < w; ix++ {
			var x, y, z float32 = 0., 0., 0.
			for iz := 0; iz < d; iz++ {
				x += arr[0][iz][iy][ix]
				y += arr[1][iz][iy][ix]
				z += arr[2][iz][iy][ix]
			}
			x /= norm
			y /= norm
			z /= norm
			img.Set(ix, (h-1)-iy, HSLMap(x, y, z))
		}
	}
	if arrowSize > 0 {
		drawArrows(img, arr, arrowSize)
	}
}

func extrema(data []float32) (min, max float32) {
	min = data[0]
	max = data[0]
	for _, d := range data {
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	return
}

// Draws rank 3 tensor (3D scalar field) as image
// averages data over X (usually thickness of thin film)
func drawFloats(img *image.RGBA, arr [][][]float32, min, max float32, colormap ...color.RGBA) {

	w, h := len(arr[0][0]), len(arr[0])
	d := len(arr)
	*img = *recycle(img, w, h)

	for iy := 0; iy < h; iy++ {
		for ix := 0; ix < w; ix++ {
			var v float32 = 0.
			for iz := 0; iz < d; iz++ {
				v += arr[iz][iy][ix]

			}
			v /= float32(d)
			img.Set(ix, (h-1)-iy, ColorMap(min, max, v, colormap...))
		}
	}
}

// recycle image if it has right size
func recycle(img *image.RGBA, w, h int) *image.RGBA {
	if img == nil || img.Bounds().Size().X != w || img.Bounds().Size().Y != h {
		img = image.NewRGBA(image.Rect(0, 0, w, h))
	}
	return img
}

const (
	X = 0
	Y = 1
	Z = 2
)
