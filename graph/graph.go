// package graph renders 2D graphs of table data
package graph

import (
	"fmt"
	"github.com/mumax/3/svgo"
	"io"
)

const (
	ML = 32
	MR = 32
	MT = 32
	MB = 32
)

func SVG(out io.Writer, wTotal, hTotal int) {

	c := svg.New(out)
	c.Start(wTotal, hTotal)

	w := wTotal - ML - MR
	h := hTotal - MT - MB

	c.Rect(ML, MR, w, h, `style="fill:none;stroke:black"`)

	c.End()
}

func hex(i uint8) string {
	return fmt.Sprintf("%02X", i)
}
