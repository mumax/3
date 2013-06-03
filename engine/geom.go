package engine

import (
	"code.google.com/p/mx3/data"
)

type geom struct {
	maskQuant
}

func (q *geom) init() {
	q.maskQuant = mask(1, Mesh(), "geom", "")
}

func (q *geom) SetFunc(f func(x, y, z float64) bool) {
	q.alloc()

	h := hostBuf(1, q.Mesh())
	l := h.Scalars()

	n := q.Mesh().Size()
	c := q.Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx
				inside := f(x, y, z)
				if inside {
					l[i][j][k] = 1
				} else {
					l[i][j][k] = 0
				}
			}
		}
	}

	data.Copy(q.buffer, h)
	//magnetization.stencil(h) ...
}

func HalfSpace() func(x, y, z float64) bool {
	return func(x, y, z float64) bool {
		return x > 0
	}
}

func Ellipsoid(rx, ry, rz float64) func(x, y, z float64) bool {
	return func(x, y, z float64) bool {
		return sqr64(x/rx)+sqr64(y/ry)+sqr64(z/rz) <= 1
	}
}

func sqr64(x float64) float64 { return x * x }
