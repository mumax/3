package engine

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func (g *guistate) servePlot(w http.ResponseWriter, r *http.Request) {

	out := []byte{}

	// handle error and return wheter err != nil.
	handle := func(err error) bool {
		if err != nil {
			w.Write(emptyIMG())
			g.Set("plotErr", "Plot error: "+err.Error()+string(out))
			return true
		} else {
			return false
		}
	}

	data, err := Table.Read()
	if handle(err) {
		return
	}

	xColIdx, err := strconv.Atoi(strings.TrimSpace(g.StringValue("usingx")))
	if err != nil || xColIdx < 0 || xColIdx >= len(data[0]) {
		handle(errors.New("Invalid column index"))
		return
	}

	yColIdx, err := strconv.Atoi(strings.TrimSpace(g.StringValue("usingy")))
	if err != nil || yColIdx < 0 || yColIdx >= len(data[0]) {
		handle(errors.New("Invalid column index"))
		return
	}

	p, err := plot.New()
	if handle(err) {
		return
	}

	header := Table.Header()

	p.X.Label.Text = header[xColIdx].Name()
	if unit := header[xColIdx].Unit(); unit != "" {
		p.X.Label.Text += " (" + unit + ")"
	}

	p.Y.Label.Text = header[yColIdx].Name()
	if unit := header[yColIdx].Unit(); unit != "" {
		p.Y.Label.Text += " (" + unit + ")"
	}

	p.X.Label.Padding = 0.2 * vg.Inch
	p.Y.Label.Padding = 0.2 * vg.Inch

	nPoints := len(data)
	points := make(plotter.XYs, nPoints)
	for i := 0; i < nPoints; i++ {
		points[i].X = data[i][xColIdx]
		points[i].Y = data[i][yColIdx]
	}

	lpLine, lpPoints, err := plotter.NewLinePoints(points)
	if handle(err) {
		return
	}
	lpLine.Color = color.RGBA{R: 255, G: 150, B: 150, A: 255}
	lpLine.Width = 2
	lpPoints.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
	lpPoints.Shape = draw.CircleGlyph{}
	lpPoints.Radius = 2

	p.Add(lpLine, lpPoints)

	wr, err := p.WriterTo(6*vg.Inch, 4*vg.Inch, "svg")
	if handle(err) {
		return
	}

	w.Header().Set("Content-Type", "image/svg+xml")
	_, err = wr.WriteTo(w)
	if handle(err) {
		return
	}

	g.Set("plotErr", "")
	return
}

var empty_img []byte

// empty image to show if there's no plot...
func emptyIMG() []byte {
	if empty_img == nil {
		o := bytes.NewBuffer(nil)
		png.Encode(o, image.NewNRGBA(image.Rect(0, 0, 4, 4)))
		empty_img = o.Bytes()
	}
	return empty_img
}
