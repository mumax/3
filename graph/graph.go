// package graph renders 2D graphs of table data
package graph

import (
	"fmt"
	"github.com/mumax/3/svgo"
	"io"
)

type Graph struct {
	canvas                          *svg.SVG
	w, h                            float64 // overall canvas size in pixels
	xMin, xMax                      float64 // x ranges of graphing area
	yMin, yMax                      float64 // y ranges of graphing area
	marginL, marginR                float64 // whitespace around graphing area, in pixels
	marginB, marginT                float64 // whitespace around graphing area, in pixels
	LineStyle, FillStyle, TextStyle string  //
	TicSize                         float64 // length of axis tics, in pixels
}

func New(out io.Writer, width, height int) *Graph {
	canvas := svg.New(out)
	canvas.Start(width, height)
	g := &Graph{canvas: canvas, w: float64(width), h: float64(height)}
	g.SetMargins(48, 24, 52, 16) // suited margin for DrawAxes
	g.SetRanges(0, 1, 0, 1)
	g.LineStyle = `style="fill:none;stroke:black"`
	g.FillStyle = `style="fill:black"`
	g.TextStyle = `font-family="sans-serif"`
	g.TicSize = 8
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

func (g *Graph) DrawAxes(xTic, yTic float64) {
	const vpad = 4 // pixels between axes and text
	g.Rect(g.xMin, g.yMin, g.xMax, g.yMax)
	for x := g.xMin; x <= g.xMax; x += xTic {
		x2, y2 := g.transf(x, g.yMax)
		g.canvas.Line(x2, y2, x2, y2+g.TicSize, g.LineStyle)
		x2, y2 = g.transf(x, g.yMin)
		g.canvas.Line(x2, y2, x2, y2-g.TicSize, g.LineStyle)
		g.canvas.Text(int(x2), int(y2)+vpad, fmt.Sprintf("%.5g", x), g.TextStyle, `dominant-baseline="hanging"`, `text-anchor="middle"`)
	}
	for y := g.yMin; y <= g.yMax; y += yTic {
		x2, y2 := g.transf(g.xMin, y)
		g.canvas.Line(x2, y2, x2+g.TicSize, y2, g.LineStyle)
		g.canvas.Text(int(x2)-vpad, int(y2), fmt.Sprintf("%.5g", y), g.TextStyle, `dominant-baseline="middle"`, `style="text-anchor:end"`)
		x2, y2 = g.transf(g.xMax, y)
		g.canvas.Line(x2, y2, x2-g.TicSize, y2, g.LineStyle)
	}
}

func (g *Graph) DrawXLabel(text string) {
	const vpad = 32 // pixels between x-axis and x label
	_, y2 := g.transf(0, g.yMin)
	g.canvas.Text(int(g.w/2), int(y2)+vpad, text, g.TextStyle, `dominant-baseline="hanging"`, `style="text-anchor:middle"`)
}

func (g *Graph) Polyline(x []float64, y []float64) {
	x2, y2 := make([]float64, len(x)), make([]float64, len(y))
	for i := range x {
		x2[i], y2[i] = g.transf(x[i], y[i])
	}
	g.canvas.Polyline(x2, y2, g.LineStyle)
}

func (g *Graph) Line(x1, y1, x2, y2 float64) {
	x := []float64{x1, x2}
	y := []float64{y1, y2}
	g.Polyline(x, y)
}

// Draw rectangle given its diagonal.
func (g *Graph) Rect(x1, y1, x2, y2 float64) {
	x := []float64{x1, x1, x2, x2, x1}
	y := []float64{y1, y2, y2, y1, y1}
	g.Polyline(x, y)
}

func (g *Graph) transf(x, y float64) (x2, y2 float64) {
	x2 = g.marginL + (x-g.xMin)*(g.w-g.marginL-g.marginR)/(g.xMax-g.xMin)
	y2 = g.h - g.marginB - (y-g.yMin)*(g.h-g.marginB-g.marginT)/(g.yMax-g.yMin)
	return
}
