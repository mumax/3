package cuda

import "code.google.com/p/mx3/data"

// 6-neighbor exchange field.
func NewExchange6(mesh *data.Mesh, aex_reduced float64) *Stencil3D {
	Δ := mesh.CellSize()
	var w [3]float32
	for i := range w {
		if mesh.Size()[i] != 1 {
			w[i] = float32(2 * aex_reduced / (Δ[i] * Δ[i]))
		}
	}
	Σ := float32(-2 * (w[0] + w[1] + w[2]))
	nabla := [7]float32{Σ, w[0], w[0], w[1], w[1], w[2], w[2]}
	s := new(Stencil3D)
	s.Weight[0][0] = nabla
	s.Weight[1][1] = nabla
	s.Weight[2][2] = nabla
	return s
}
