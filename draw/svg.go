package draw

import (
	"github.com/ajstarks/svgo"
	"io"
)

func SVG(out io.Writer, arr [3][][][]float32) {

	h, w := len(arr[0][0]), len(arr[0][0][0])

	canvas := svg.New(out)
	canvas.Start(w, h)
	canvas.Circle(w/2, h/2, 100)
	canvas.Text(w/2, h/2, "Hello, SVG", "text-anchor:middle;font-size:30px;fill:white")
	canvas.End()
}
