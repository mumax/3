package draw

import (
	"fmt"
	"io"
	"math"

	svg "github.com/mumax/3/svgo"
)

// Renders svg image of vector data.
func SVG(out io.Writer, arr [3][][][]float32) {

	h, w := len(arr[0][0]), len(arr[0][0][0])

	const (
		r1 = 1. / 2. // arrow half length
		r2 = 1. / 4. // arrow half width
	)

	canvas := svg.New(out)
	canvas.Start(w, h)

	for slice := 0; slice < len(arr[0]); slice++ {
		Mx := arr[X][slice]
		My := arr[Y][slice]
		Mz := arr[Z][slice]

		for i := range h {
			y := float64(h) - (float64(i) + 1./2.)
			for j := range w {
				x := float64(j) + 1./2.

				mx := Mx[i][j]
				my := My[i][j]
				mz := Mz[i][j]

				// skip zero-length vectors
				if mx*mx+my*my+mz*mz == 0 {
					continue
				}

				theta := math.Atan2(float64(my), float64(mx))
				c := math.Cos(theta)
				s := math.Sin(theta)
				r1 := r1 * math.Cos(math.Asin(float64(mz)))

				xs := []float64{(r1 * c) + x, (r2*s - r1*c) + x, (-r2*s - r1*c) + x}
				ys := []float64{-(r1 * s) + y, -(-r2*c - r1*s) + y, -(r2*c - r1*s) + y}

				col := HSLMap(mx, my, mz)
				style := "fill:#" + hex(col.R) + hex(col.G) + hex(col.B)

				canvas.Polygon(xs, ys, style)
			}
		}
	}

	canvas.End()
}

func hex(i uint8) string {
	j := max(int(i)-32, 0) // make it a bit darker
	return fmt.Sprintf("%02X", j)
}
