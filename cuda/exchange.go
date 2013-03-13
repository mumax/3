package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"github.com/barnex/cuda5/cu"
)

// TODO: m should be normalized.
func Exchange(Hex *data.Slice, M *data.Slice, Aex float64) {
	// TODO: size check
	mesh := Hex.Mesh()
	N := mesh.Size()
	c := mesh.CellSize()
	Aex *= 2 * mag.Mu0
	w0 := float32(Aex / (c[0] * c[0]))
	w1 := float32(Aex / (c[1] * c[1]))
	w2 := float32(Aex / (c[2] * c[2]))
	cfg := make2DConf(N[2], N[1])

	str := [3]cu.Stream{stream(), stream(), stream()}
	for c := 0; c < 3; c++ {
		k_exchange1comp_async(Hex.DevPtr(c), M.DevPtr(c), w0, w1, w2, N[0], N[1], N[2], cfg, str[c])
	}
	syncAndRecycle(str[0])
	syncAndRecycle(str[1])
	syncAndRecycle(str[2])

	//	k_exchange(Hex.DevPtr(0), Hex.DevPtr(1), Hex.DevPtr(2),
	//		M.DevPtr(0), M.DevPtr(1), M.DevPtr(2),
	//		w0, w1, w2, N[0], N[1], N[2], cfg)
}
