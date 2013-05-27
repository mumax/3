package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Stores values defined on the faces in-between cells.
// E.g.: the exchange interaction between cells.
// Automatically initialized to all ones.
type staggeredMaskQuant struct {
	maskQuant
}

func newStaggeredMask(m *data.Mesh, name, unit string) *staggeredMaskQuant {
	return &staggeredMaskQuant{*newMask(3, m, name, unit)}
}

// Sets the value at the "lower/left" face of cell(ix, iy, iz). E.g.:
// 	SetSide1(X, i, j, k, value) // sets value on the left side of cell i,j,k.
// 	SetSide1(Y, i, j, k, value) // sets value below (along Y) of cell i,j,k.
// 	SetSide1(Z, i, j, k, value) // sets value below (along Z) of cell i,j,k.
func (m *staggeredMaskQuant) SetSide1(direction int, ix, iy, iz int, value float64) {
	m.init()
	cuda.SetCell(m.buffer, util.SwapIndex(direction, 3), iz, iy, ix, float32(value))
}

// Sets the value at the "upper/right" face of cell(ix, iy, iz). E.g.:
// 	SetSide2(X, i, j, k, value) // sets value on the right side of cell i,j,k.
// 	SetSide2(Y, i, j, k, value) // sets value above (along Y) of cell i,j,k.
// 	SetSide2(Z, i, j, k, value) // sets value above (along Z) of cell i,j,k.
func (m *staggeredMaskQuant) SetSide2(direction int, ix, iy, iz int, value float64) {
	m.init()
	direction = util.SwapIndex(direction, 3)
	i := [3]int{iz, iy, ix}
	i[direction]++
	size := m.buffer.Mesh().Size()
	if i[direction] == size[direction] {
		i[direction] = 0 // wrap around boundary
	}
	cuda.SetCell(m.buffer, direction, i[0], i[1], i[2], float32(value))
}
