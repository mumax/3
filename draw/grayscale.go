package draw

import "image/color"

// Gray colorscale: min=black, max=white.
func GreyMap(min, max, value float32) color.RGBA {
	col := (value - min) / (max - min)
	if col > 1. {
		col = 1.
	}
	if col < 0. {
		col = 0.
	}
	color8 := uint8(255 * col)
	return color.RGBA{color8, color8, color8, 255}
}
