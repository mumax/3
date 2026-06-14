package draw

import (
	"image/color"
	"math"
)

// Colormap for 3D vector data.
func HSLMap(x, y, z float32) color.RGBA {
	s := sqrtf(x*x + y*y + z*z)
	l := 0.5*z + 0.5
	h := float32(math.Atan2(float64(y), float64(x)))
	return HSLtoRGB(h, s, l)
}

// h = 0..2pi, s=0..1, l=0..1
func HSLtoRGB(h, s, l float32) color.RGBA {
	if s > 1 {
		s = 1
	}
	if l > 1 {
		l = 1
	}

	h = h * (180.0 / math.Pi / 60.0)
	for h < 0 {
		h += 6
	}
	for h >= 6 {
		h -= 6
	}

	var c float32 // chroma
	if l <= 0.5 {
		c = 2 * l * s
	} else {
		c = (2 - 2*l) * s
	}
	x := c * (1 - abs(fmod(h, 2)-1))

	var r, g, b float32

	switch {
	case 0 <= h && h < 1:
		r, g, b = c, x, 0.
	case 1 <= h && h < 2:
		r, g, b = x, c, 0.
	case 2 <= h && h < 3:
		r, g, b = 0., c, x
	case 3 <= h && h < 4:
		r, g, b = 0, x, c
	case 4 <= h && h < 5:
		r, g, b = x, 0., c
	case 5 <= h && h < 6:
		r, g, b = c, 0., x
	}

	m := l - 0.5*c
	r, g, b = r+m, g+m, b+m
	R, G, B := uint8(255*r), uint8(255*g), uint8(255*b)
	return color.RGBA{R, G, B, 255}
}

// modulo
func fmod(number, mod float32) float32 {
	for number < mod {
		number += mod
	}
	for number >= mod {
		number -= mod
	}
	return number
}

func abs(number float32) float32 {
	if number < 0 {
		return -number
	} // else
	return number
}

func sqrtf(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
