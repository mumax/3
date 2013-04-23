package cuda

import (
	"code.google.com/p/mx3/data"
	"github.com/barnex/cuda5/cu"
)

//#include "stencil.h"
import "C"

const STENCIL_BLOCKSIZE = C.STENCIL_BLOCKSIZE_X

// Add exchange field to Beff.
func AddExchange(Beff, m, mask *data.Slice, Aex, Msat float64) {
	// TODO: size check
	if mask == nil {
		AddAnisoExchange(Beff, m, Aex, Aex, Aex, Msat)
	} else {
		AddMaskExchange(Beff, m, mask, Aex, Msat)
	}
}

// Add exchange field to Beff with different exchange constant for X,Y,Z direction.
// m must be normalized to unit length.
func AddAnisoExchange(Beff, m *data.Slice, AexX, AexY, AexZ, Msat float64) {

	mesh := Beff.Mesh()
	N := mesh.Size()
	w := exchangeWeights(AexX, AexY, AexZ, Msat, mesh.CellSize())
	cfg := make2DConfSize(N[2], N[1], STENCIL_BLOCKSIZE)

	str := [3]cu.Stream{stream(), stream(), stream()}
	for c := 0; c < 3; c++ {
		k_addexchange1comp_async(Beff.DevPtr(c), m.DevPtr(c), w[0], w[1], w[2], N[0], N[1], N[2], cfg, str[c])
	}
	syncAndRecycle(str[0])
	syncAndRecycle(str[1])
	syncAndRecycle(str[2])
}

// mask defines a pre-factor for (Aex / Msat) to allow space-dependent exchange.
// the mask is staggered over half a cell with respect to the magnetization grid,
// and otherwise has the same size.
// mask{X,Y,Z} defines the coupling between neighbors in the {X,Y,Z} direction, respectively.
// 	mask[X][i, j, k] defines the coupling between m[i,  j, ,k-1] and m[i, j, k]
// 	mask[Y][i, j, k] defines the coupling between m[i,  j-1,k  ] and m[i, j, k]
// 	mask[Z][i, j, k] defines the coupling between m[i-1,j, ,k  ] and m[i, j, k]
// Each time, the 0th element defines the coupling at the leftmost boundary and is thus unused,
// but would be used in case if periodic boundary conditions.
func AddMaskExchange(Beff, m, mask *data.Slice, Aex, Msat float64) {

	mesh := Beff.Mesh()
	N := mesh.Size()
	w := exchangeWeights(Aex, Aex, Aex, Msat, mesh.CellSize())
	cfg := make2DConfSize(N[2], N[1], STENCIL_BLOCKSIZE)

	str := [3]cu.Stream{stream(), stream(), stream()}
	for c := 0; c < 3; c++ {
		k_addexchangemask_async(Beff.DevPtr(c), m.DevPtr(c),
			mask.DevPtr(0), mask.DevPtr(1), mask.DevPtr(2),
			w[0], w[1], w[2], N[0], N[1], N[2], cfg, str[c])
	}
	syncAndRecycle(str[0])
	syncAndRecycle(str[1])
	syncAndRecycle(str[2])
}

func exchangeWeights(AexX, AexY, AexZ, Msat float64, c [3]float64) [3]float32 {
	w0 := float32(2 * AexX / (Msat * c[0] * c[0]))
	w1 := float32(2 * AexY / (Msat * c[1] * c[1]))
	w2 := float32(2 * AexZ / (Msat * c[2] * c[2]))
	return [3]float32{w0, w1, w2}
}
