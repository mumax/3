package cuda

import (
	"code.google.com/p/mx3/data"
	"github.com/barnex/cuda5/cu"
)

//#include "stencil.h"
import "C"

const STENCIL_BLOCKSIZE = C.STENCIL_BLOCKSIZE_X

// Add exchange field to Beff.
func AddExchange(Beff *data.Slice, m *data.Slice, Aex, Msat float64) {
	AddAnisoExchange(Beff, m, Aex, Aex, Aex, Msat)
}

// Add exchange field to Beff with different exchange constant for X,Y,Z direction.
// m must be normalized to unit length.
func AddAnisoExchange(Beff *data.Slice, m *data.Slice, AexX, AexY, AexZ, Msat float64) {
	// TODO: size check
	mesh := Beff.Mesh()
	N := mesh.Size()
	c := mesh.CellSize()
	w0 := float32(2 * AexX / (Msat * c[0] * c[0]))
	w1 := float32(2 * AexY / (Msat * c[1] * c[1]))
	w2 := float32(2 * AexZ / (Msat * c[2] * c[2]))
	cfg := make2DConfSize(N[2], N[1], STENCIL_BLOCKSIZE)

	str := [3]cu.Stream{stream(), stream(), stream()}
	for c := 0; c < 3; c++ {
		k_addexchange1comp_async(Beff.DevPtr(c), m.DevPtr(c), w0, w1, w2, N[0], N[1], N[2], cfg, str[c])
	}
	syncAndRecycle(str[0])
	syncAndRecycle(str[1])
	syncAndRecycle(str[2])
}
