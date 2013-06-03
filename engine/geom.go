package engine

import (
	"code.google.com/p/mx3/data"
)

// geometry mask for the magnetization
type geomMask struct {
	maskQuant
	host *data.Slice // host copy of maskQuant
}

func (g *geomMask) init() {
	g.maskQuant = mask(1, Mesh(), "geom", "")
}

// set the mask to 1 where f is true, 0 elsewhere
func (g *geomMask) Rasterize(f Shape) {
	g.alloc()
	if g.host == nil {
		g.host = hostBuf(1, g.Mesh())
	}

	l := g.host.Scalars()

	n := g.Mesh().Size()
	c := g.Mesh().CellSize()
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

	data.Copy(g.buffer, g.host)
	M.stencil(g.host)
}
