package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"github.com/barnex/cuda5/cu"
)

// Add exchange field to Beff. m must be normalized to unit length.
func AddExchange(Beff *data.Slice, m *data.Slice, Aex, Bsat float64) {
	// TODO: size check
	mesh := Beff.Mesh()
	N := mesh.Size()
	c := mesh.CellSize()
	fac := 2 * Aex * mag.Mu0 / Bsat
	w0 := float32(fac / (c[0] * c[0]))
	w1 := float32(fac / (c[1] * c[1]))
	w2 := float32(fac / (c[2] * c[2]))
	cfg := make2DConf(N[2], N[1])

	str := [3]cu.Stream{stream(), stream(), stream()}
	for c := 0; c < 3; c++ {
		k_addexchange1comp_async(Beff.DevPtr(c), m.DevPtr(c), w0, w1, w2, N[0], N[1], N[2], cfg, str[c])
	}
	syncAndRecycle(str[0])
	syncAndRecycle(str[1])
	syncAndRecycle(str[2])
}
