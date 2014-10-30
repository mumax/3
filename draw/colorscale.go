package draw

import "image/color"
import "fmt"

func ColorMap(min, max, value float32, colormap ...color.RGBA) color.RGBA {
	if len(colormap) == 0 {
		colormap = []color.RGBA{{0, 0, 0, 255}, {255, 255, 255, 255}}
	}

	col := (value - min) / (max - min)
	if col > 1. {
		col = 1.
	}
	if col < 0. {
		col = 0.
	}

	N := float64(len(colormap)) - 1
	lower := int(float64(col) * N)
	upper := lower + 1

	if lower < 0 {
		panic(fmt.Sprint("lower=", lower))
	}

	if upper >= len(colormap) {
		return colormap[lower]
	}

	c1 := colormap[lower]
	c2 := colormap[upper]

	x := (float64(col) - float64(lower)/N)

	fmt.Println(x)

	if x < 0 || x > 1 {
		panic(fmt.Sprint("x=", x))
	}

	r := (1-x)*float64(c1.R) + x*float64(c2.R)
	g := (1-x)*float64(c1.G) + x*float64(c2.G)
	b := (1-x)*float64(c1.B) + x*float64(c2.B)
	a := (1-x)*float64(c1.A) + x*float64(c2.A)

	return color.RGBA{bte(r), bte(g), bte(b), bte(a)}
}

func bte(x float64) uint8 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 255
	}
	return uint8(255 * x)
}
