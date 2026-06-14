package draw

import "image/color"
import "fmt"

type ColorMapSpec struct {
	Cmap  []color.RGBA
	Ccomp int
}

func ColorMap(min, max, value float32, colormap ...color.RGBA) color.RGBA {
	// default colormap: black-white
	if len(colormap) < 1 {
		colormap = []color.RGBA{{0, 0, 0, 255}, {255, 255, 255, 255}}
	}

	// map value to interval [O,1]
	val := float64((value - min) / (max - min))
	if val > 1 {
		val = 1
	}
	if val < 0 {
		val = 0
	}

	// find index of color below our value
	maxIndex := float64(len(colormap) - 1)
	index := val * maxIndex
	// corner case val==max:
	if index == maxIndex {
		index--
	}

	// get two neighboring colors
	i := int(index)
	if i < 0 {
		i = 0
	}
	if i >= len(colormap)-1 {
		i = len(colormap) - 2
	}
	c1 := colormap[i]
	c2 := colormap[i+1]

	// location between two neighboring colors [0..1]
	x := (val - float64(i)/maxIndex) * maxIndex
	if x < 0 || x > 1 {
		panic(fmt.Sprint("x=", x))
	}

	// interpolate between colors
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
	if x > 255 {
		return 255
	}
	return uint8(x)
}
