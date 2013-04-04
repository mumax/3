package main

// Image output.
// Author: Arne Vansteenkiste

import (
	"bufio"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/draw"
	"code.google.com/p/mx3/util"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"strconv"
)

func dumpPNG(out io.Writer, f *data.Slice) error {
	img := render(f)
	buf := bufio.NewWriter(out)
	defer buf.Flush()
	return png.Encode(buf, img)
}

func dumpJPEG(out io.Writer, f *data.Slice) error {
	img := render(f)
	buf := bufio.NewWriter(out)
	defer buf.Flush()
	return jpeg.Encode(buf, img, &jpeg.Options{Quality: 100})
}

func render(f *data.Slice) *image.NRGBA {
	dim := f.NComp()
	switch dim {
	default:
		log.Fatalf("unsupported number of components: %v", dim)
	case 3:
		return drawVectors(f.Vectors())
	case 1:
		min, max := extrema(f.Host()[0])
		if *flag_min != "auto" {
			m, err := strconv.ParseFloat(*flag_min, 32)
			util.FatalErr(err)
			min = float32(m)
		}
		if *flag_max != "auto" {
			m, err := strconv.ParseFloat(*flag_max, 32)
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
			img.Set(j, (h-1)-i, draw.HSLMap(z, y, x))
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
			img.Set(j, (h-1)-i, draw.GreyMap(min, max, x))
		}
	}
	return img
}
