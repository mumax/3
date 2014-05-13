// package graph renders 2D graphs of table data
package graph

import (
	"github.com/mumax/3/svgo"
)

const S = 256 // fixed-point scaling

func SVG(out io.Writer, w, h int) {

	canvas := svg.New(out)
	canvas.Start(S*w, S*h)

	canvas.End()
}

func hex(i uint8) string {
	return fmt.Sprintf("%02X", j)
}
