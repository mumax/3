package draw

import (
	"code.google.com/p/freetype-go/freetype/raster"
	"github.com/mumax/3/data"
	"image"
	"image/color"
	"math"
)

func drawArrows(img *image.RGBA, arr [3][][][]float32, sub int) {
	c := NewCanvas(img)

	Na := data.SizeOf(arr[0]) // number of arrows
	Na[X] = imax(Na[X]/sub, 1)
	Na[Y] = imax(Na[Y]/sub, 1)
	Na[Z] = 1
	small := data.Downsample(arr[:], Na)
	S := float32(sub)

	for iy := 0; iy < Na[Y]; iy++ {
		Ay := (float32(iy) + 0.5) * S
		for ix := 0; ix < Na[X]; ix++ {
			Ax := (float32(ix) + 0.5) * S
			mx := small[X][0][iy][ix]
			my := small[Y][0][iy][ix]
			mz := small[Z][0][iy][ix]
			//col := HSLMap(mx, my, mz)
			//col.R /= 2
			//col.G /= 2
			//col.B /= 2
			//col.A = 128
			col := color.RGBA{0, 0, 0, 128}
			c.SetColor(col)
			c.Arrow(Ax, Ay, mx, my, mz)

		}
	}
}

// A Canvas is used to draw on.
type Canvas struct {
	*image.RGBA
	*raster.RGBAPainter
	rasterizer  *raster.Rasterizer
	strokewidth raster.Fix32
	strokecap   raster.Capper
	path        raster.Path
}

// Make a new canvas of size w x h.
func NewCanvas(img *image.RGBA) *Canvas {
	c := new(Canvas)
	c.RGBA = img
	c.RGBAPainter = raster.NewRGBAPainter(c.RGBA)
	c.rasterizer = raster.NewRasterizer(img.Bounds().Max.X, img.Bounds().Max.Y)
	c.rasterizer.UseNonZeroWinding = true
	c.SetColor(color.Black)
	c.path = make(raster.Path, 0, 100)
	c.resetPath()
	c.SetStroke(1, raster.RoundCapper)
	return c
}

func (c *Canvas) Arrow(x, y, mx, my, mz float32) {

	const (
		ln = 7
		r2 = 3
	)

	theta := math.Atan2(float64(my), float64(mx))
	cos := float32(math.Cos(theta))
	sin := float32(math.Sin(theta))
	r1 := ln * float32(math.Cos(math.Asin(float64(mz))))

	pt1 := pt((r1*cos)+x, (r1*sin)+y)
	pt2 := pt((r2*sin-r1*cos)+x, (-r2*cos-r1*sin)+y)
	pt3 := pt((-r2*sin-r1*cos)+x, (r2*cos-r1*sin)+y)

	c.resetPath()
	c.path.Start(pt1)
	c.path.Add1(pt2)
	c.path.Add1(pt3)
	c.path.Add1(pt1)
	c.fillPath()
}

// Set the line width and end capping style.
func (c *Canvas) SetStroke(width float32, cap_ raster.Capper) {
	c.strokewidth = fix32(width)
	c.strokecap = cap_
}

func (c *Canvas) resetPath() {
	c.path = c.path[:0]
}

func (c *Canvas) strokePath() {
	raster.Stroke(c.rasterizer, c.path, c.strokewidth, c.strokecap, nil)
	c.rasterizer.Rasterize(c.RGBAPainter)
}

func (c *Canvas) fillPath() {
	c.rasterizer.AddPath(c.path)
	c.rasterizer.Rasterize(c.RGBAPainter)
}

func pt(x, y float32) raster.Point {
	return raster.Point{fix32(x), fix32(y)}
}

func fix32(x float32) raster.Fix32 {
	return raster.Fix32(int(x * (1 << 8)))
}

func imax(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}
