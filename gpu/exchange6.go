package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
)

func NewExchange6(tag string, m nimble.ChanN, aex_reduced float64) *Stencil3D {
	Δ := m.Mesh().CellSize()
	var w [3]float32
	for i := range w {
		if m.Mesh().Size()[i] != 1 {
			w[i] = float32(2 * aex_reduced / (Δ[i] * Δ[i]))
		}
	}
	Σ := float32(-2 * (w[0] + w[1] + w[2]))
	nabla := [7]float32{Σ, w[0], w[0], w[1], w[1], w[2], w[2]}
	s := NewStencil3D(tag, "T", m)
	s.Weight[0][0] = nabla
	s.Weight[1][1] = nabla
	s.Weight[2][2] = nabla
	return s
}
