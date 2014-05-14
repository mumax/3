// package graph renders 2D graphs of table data
package graph

import (
	"github.com/mumax/3/svgo"
	"io"
)

type Graph struct {
	canvas           *svg.SVG
	w, h             float64
	xMin, xMax       float64
	yMin, yMax       float64
	marginL, marginR float64
	marginB, marginT float64
	LineStyle        string
}

func New(out io.Writer, width, height int) *Graph {
	canvas := svg.New(out)
	canvas.Start(width, height)
	g := &Graph{canvas: canvas, w: float64(width), h: float64(height)}
	g.SetMargins(width/16, width/16, height/16, height/16)
	g.SetRanges(0, 1, 0, 1)
	g.LineStyle = `style="stroke:black"`
	return g
}

func (g *Graph) End() {
	g.canvas.End()
}

func (g *Graph) SetMargins(left, right, bottom, top int) {
	g.marginL = float64(left)
	g.marginR = float64(right)
	g.marginB = float64(bottom)
	g.marginT = float64(top)
}

func (g *Graph) SetRanges(xMin, xMax, yMin, yMax float64) {
	g.xMin = xMin
	g.xMax = xMax
	g.yMin = yMin
	g.yMax = yMax
}

func (g *Graph) Line(x1, y1, x2, y2 float64) {
	x1, y1 = g.transf(x1, y1)
	x2, y2 = g.transf(x2, y2)
	g.canvas.Line(x1, y1, x2, y2, g.LineStyle)
}

func (g *Graph) transf(x, y float64) (x2, y2 float64) {
	x2 = g.marginL + x*(g.w-g.marginL-g.marginR)/(g.xMax-g.xMin)
	y2 = g.h - g.marginT - y*(g.h-g.marginB-g.marginT)/(g.yMax-g.yMin)
	return
}
