package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
	"code.google.com/p/mx3/mag"
)

func Exchange(Hex *data.Slice, M *data.Slice, Aex float64) {
	// TODO: size check
	mesh := Hex.Mesh()
	N := mesh.Size()
	c := mesh.CellSize()
	Aex *= mag.Mu0
	w0 := float32(Aex / (c[0] * c[0]))
	w1 := float32(Aex / (c[1] * c[1]))
	w2 := float32(Aex / (c[2] * c[2]))
	gr, bl := Make2DConf(N[2], N[1])
	kernel.K_exchange(Hex.DevPtr(0), Hex.DevPtr(1), Hex.DevPtr(2),
		M.DevPtr(0), M.DevPtr(1), M.DevPtr(2),
		w0, w1, w2, N[0], N[1], N[2], gr, bl)
}
