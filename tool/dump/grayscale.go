package main

import (
	"image"
	"image/color"
)

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
			img.Set(j, (h-1)-i, GreyMap(min, max, x))
		}
	}
	return img
}

func GreyMap(min, max, value float32) color.NRGBA {
	col := (value - min) / (max - min)
	if col > 1. {
		col = 1.
	}
	if col < 0. {
		col = 0.
	}
	color8 := uint8(255 * col)
	return color.NRGBA{color8, color8, color8, 255}
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
