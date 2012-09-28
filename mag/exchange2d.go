package mag

import "nimble-cube/core"

func NewExchange2D(m core.RChan3, hex core.Chan3, mesh *core.Mesh, aex_reduced float64) *Stencil2D {
	cellsize := mesh.CellSize()
	s := NewStencil2D(m, hex)
	J := float32(aex_reduced / (cellsize[1] * cellsize[1]))
	K := float32(aex_reduced / (cellsize[2] * cellsize[2]))
	s.W00 = -2*J - 2*K
	s.W_10, s.W10 = J, J
	s.W0_1, s.W01 = K, K
	return s
}
