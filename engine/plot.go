package engine

import (
	"fmt"
	"github.com/mumax/3/graph"
	"github.com/mumax/3/svgo"
	"math"
	"net/http"
)

func (g *guistate) servePlot(w http.ResponseWriter, r *http.Request) {

	// atomically get local copies to avoid race conditions
	var (
		data [][][]float64
	)
	InjectAndWait(func() {
		hist := Table.history
		data = make([][][]float64, len(hist))
		for i := range data {
			data[i] = make([][]float64, len(hist[i]))
			copy(data[i], hist[i]) // only copies slice headers, sufficient to avoid races (append to slices only)
		}
	})

	w.Header().Set("Content-Type", "image/svg+xml")

	if len(data) == 0 || len(data[0]) == 0 || len(data[0][0]) < 2 { // need 2 points (from time column)
		// send a small empty plot as placeholder
		plot := svg.New(w)
		plot.Start(600, 64)
		plot.Text(0, 32, "Enable with TableAutosave(interval)")
		plot.End()
		return
	}
	plot := graph.New(w, 600, 300)
	tMax := data[0][0][len(data[0][0])-1]
	tMax = roundNice(tMax)
	plot.SetRanges(0, tMax, -1, 1)
	plot.DrawAxes(tMax/5, 0.5)
	plot.DrawXLabel("t (s)")
	for c := 0; c < 3; c++ {
		plot.LineStyle = fmt.Sprintf(`style="fill:none;stroke-width:2;stroke:%v"`, [3]string{"red", "green", "blue"}[c])
		plot.Polyline(data[0][0], data[1][c])
	}
	plot.End()
}

// round (up) to nice 2 digit number for max axis range.
func roundNice(x float64) float64 {
	order := int(math.Log10(x))
	y := math.Ceil(1E2 * x / math.Pow10(order)) // 1E2: 2 digits
	return y * math.Pow10(order) / 1E2
}
