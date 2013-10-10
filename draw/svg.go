package draw

import (
	"github.com/ajstarks/svgo"
	"io"
	"math"
)

func SVG(out io.Writer, arr [3][][][]float32) {

	slice := 0 // todo

	Mx := arr[2][slice]
	My := arr[1][slice]
	//Mz := arr[0][slice]

	h, w := len(Mx), len(Mx[0])

	const (
		S  = 100 // scale
		r1 = S / 2
		r2 = S / 4
	)

	canvas := svg.New(out)
	canvas.Start(S*w, S*h)

	for i := 0; i < h; i++ {
		y := S*i + S/2
		for j := 0; j < w; j++ {
			x := S*j + S/2

			mx := Mx[i][j]
			my := My[i][j]
			theta := math.Atan2(float64(my), float64(mx))
			c := math.Cos(theta)
			s := math.Sin(theta)

			xs := []int{int(r1*c) + x, int(r2*s-r1*c) + x, int(-r2*s-r1*c) + x}
			ys := []int{int(r1*s) + y, int(-r2*c-r1*s) + y, int(r2*c-r1*s) + y}

			canvas.Polygon(xs, ys)
		}
	}

	canvas.End()
}
