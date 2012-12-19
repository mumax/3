package main

// Image output.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/draw"
	"code.google.com/p/mx3/dump"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path"
	"strconv"
)

func dumpImage(f *dump.Frame, file string) {
	var img *image.NRGBA
	{
		dim := f.NComp()
		switch dim {
		default:
			core.Fatal(fmt.Errorf("unsupported number of components: %v", dim))
		case 3:
			img = DrawVectors(f.Vectors())
		case 1:
			min, max := extrema(f.Data)
			if *flag_min != "auto" {
				m, err := strconv.ParseFloat(*flag_min, 32)
				core.Fatal(err)
				min = float32(m)
			}
			if *flag_max != "auto" {
				m, err := strconv.ParseFloat(*flag_max, 32)
				core.Fatal(err)
				max = float32(m)
			}
			img = DrawFloats(f.Floats(), min, max)
		}
	}

	out, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	core.Fatal(err)
	defer out.Close()

	ext := path.Ext(file)
	switch ext {
	default:
		core.Fatal(fmt.Errorf("unsupported image type: %v", ext))
	case ".png":
		core.Fatal(png.Encode(out, img))
	case ".jpg":
		core.Fatal(jpeg.Encode(out, img, nil))
	}
}

// Draws rank 4 tensor (3D vector field) as image
// averages data over X (usually thickness of thin film)
func DrawVectors(arr [3][][][]float32) *image.NRGBA {
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
func DrawFloats(arr [][][]float32, min, max float32) *image.NRGBA {

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
