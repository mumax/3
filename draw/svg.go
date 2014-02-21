package draw

import (
	"fmt"
	"github.com/ajstarks/svgo"
	"io"
	"math"
)

// Renders svg image of vector data.
func SVG(out io.Writer, arr [3][][][]float32) {

	h, w := len(arr[0][0]), len(arr[0][0][0])

	const (
		S  = 256 // scale
		r1 = S / 2
		r2 = S / 4
	)

	canvas := svg.New(out)
	canvas.Start(S*w, S*h)

	for slice := 0; slice < len(arr[0]); slice++ {
		Mx := arr[X][slice]
		My := arr[Y][slice]
		Mz := arr[Z][slice]

		for i := 0; i < h; i++ {
			y := (S * h) - (S*i + S/2)
			for j := 0; j < w; j++ {
				x := S*j + S/2

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

				xs := []int{int(r1*c) + x, int(r2*s-r1*c) + x, int(-r2*s-r1*c) + x}
				ys := []int{-int(r1*s) + y, -int(-r2*c-r1*s) + y, -int(r2*c-r1*s) + y}

				col := HSLMap(mx, my, mz)
				style := "fill:#" + hex(col.R) + hex(col.G) + hex(col.B)

				canvas.Polygon(xs, ys, style)
			}
		}
	}

	canvas.End()
}

func hex(i uint8) string {
	j := int(i) - 32 // make it a bit darker
	if j < 0 {
		j = 0
	}
	return fmt.Sprintf("%02X", j)
}
