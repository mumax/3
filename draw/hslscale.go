package draw

// HSL image output for vector data.
// Author: Arne Vansteenkiste

import (
	"github.com/barnex/fmath"
	"image/color"
	"math"
)

func HSLMap(x, y, z float32) color.NRGBA {
	s := fmath.Sqrt(x*x + y*y + z*z)
	l := 0.5*z + 0.5
	h := float32(math.Atan2(float64(y), float64(x)))
	return HSL(h, s, l)
}

// h = 0..2pi, s=0..1, l=0..1
func HSL(h, s, l float32) color.NRGBA {
	if s > 1 {
		s = 1
	}
	if l > 1 {
		l = 1
	}
	for h < 0 {
		h += 2 * math.Pi
	}
	for h > 2*math.Pi {
		h -= 2 * math.Pi
	}
	h = h * (180.0 / math.Pi / 60.0)

	// chroma
	var c float32
	if l <= 0.5 {
		c = 2 * l * s
	} else {
		c = (2 - 2*l) * s
	}

	x := c * (1 - abs(fmod(h, 2)-1))

	var (
		r, g, b float32
	)

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
	default:
		r, g, b = 0., 0., 0.
	}

	m := l - 0.5*c
	r, g, b = r+m, g+m, b+m

	if r > 1. {
		r = 1.
	}
	if g > 1. {
		g = 1.
	}
	if b > 1. {
		b = 1.
	}

	if r < 0. {
		r = 0.
	}
	if g < 0. {
		g = 0.
	}
	if b < 0. {
		b = 0.
	}

	R, G, B := uint8(255*r), uint8(255*g), uint8(255*b)
	return color.NRGBA{R, G, B, 255}
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
