package draw

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"image"
	"log"
	"strconv"
)

// Renders an image of slice. fmin, fmax = "auto" or a number to set the min/max color scale.
func Image(f *data.Slice, fmin, fmax string) *image.NRGBA {
	dim := f.NComp()
	switch dim {
	default:
		log.Fatalf("unsupported number of components: %v", dim)
	case 3:
		return drawVectors(f.Vectors())
	case 1:
		min, max := extrema(f.Host()[0])
		if fmin != "auto" {
			m, err := strconv.ParseFloat(fmin, 32)
			util.FatalErr(err)
			min = float32(m)
		}
		if fmax != "auto" {
			m, err := strconv.ParseFloat(fmax, 32)
			util.FatalErr(err)
			max = float32(m)
		}
		return drawFloats(f.Scalars(), min, max)
	}
	panic("unreachable")
}

// Draws rank 4 tensor (3D vector field) as image
// averages data over X (usually thickness of thin film)
func drawVectors(arr [3][][][]float32) *image.NRGBA {
	h, w := len(arr[0][0]), len(arr[0][0][0])
	d := len(arr[0])
	norm := float32(d)
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			var x, y, z float32 = 0., 0., 0.
			for k := 0; k < d; k++ {
				x += arr[0][k][i][j]
				y += arr[1][k][i][j]
				z += arr[2][k][i][j]
			}
			x /= norm
			y /= norm
			z /= norm
			img.Set(j, (h-1)-i, HSLMap(z, y, x))
		}
	}
	return img
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
func drawFloats(arr [][][]float32, min, max float32) *image.NRGBA {

	h, w := len(arr[0]), len(arr[0][0])
	d := len(arr)
	img := image.NewNRGBA(image.Rect(0, 0, w, h))

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			var x float32 = 0.
			for k := 0; k < d; k++ {
				x += arr[k][i][j]

			}
			x /= float32(d)
			img.Set(j, (h-1)-i, GreyMap(min, max, x))
		}
	}
	return img
}
