package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
)

func NewExchange6(tag string, m nimble.ChanN, aex_reduced float64) *Stencil3D {
	Δ := m.Mesh().CellSize()
	x := float32(aex_reduced / (Δ[0] * Δ[0]))
	y := float32(aex_reduced / (Δ[1] * Δ[1]))
	z := float32(aex_reduced / (Δ[2] * Δ[2]))
	Σ := float32(-2 * (x + y + z))
	nabla := [7]float32{Σ, x, x, y, y, z, z}
	s := NewStencil3D(tag, "T", m)
	s.Weight[0][0] = nabla
	s.Weight[1][1] = nabla
	s.Weight[2][2] = nabla
	return s
}
