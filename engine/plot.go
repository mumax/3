package engine

import (
	"github.com/mumax/3/graph"
	"log"
	"net/http"
)

var (
	_graphx, _graphy []float64
)

func (g *guistate) servePlot(w http.ResponseWriter, r *http.Request) {

	log.Println("plotting...")

	// atomically get local copies to
	var (
		graphx, graphy []float64
		t              float64
	)
	InjectAndWait(func() {
		t = Time
		graphx, graphy = _graphx, _graphy // copy of slice header is enough to avoid race as long as we only append to it
	})

	w.Header().Set("Content-Type", "image/svg+xml")
	plot := graph.New(w, 600, 300) // (!) canvas size duplicated in html.go
	plot.SetRanges(0, t, -1, 1)
	plot.DrawAxes(t/5, 0.5)
	plot.DrawXLabel("t (s)")
	if len(graphx) > 1 {
		plot.Polyline(graphx, graphy)
	}
	plot.End()
}
