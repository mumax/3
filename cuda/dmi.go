package cuda

import "code.google.com/p/mx3/data"

// Add effective field of Dzyaloshinskii-Moriya interaction to Beff (Tesla).
// m: normalized
// D: J/m²
func AddDMI(Beff *data.Slice, m *data.Slice, Dx, Dy, Dz, Msat float64) {
	// TODO: size check
	mesh := Beff.Mesh()
	N := mesh.Size()
	c := mesh.CellSize()

	dx := float32(2 * Dx / (Msat * c[0]))
	dy := float32(2 * Dy / (Msat * c[1]))
	dz := float32(2 * Dz / (Msat * c[2]))

	cfg := make2DConf(N[2], N[1])
	k_adddmi(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		dx, dy, dz, N[0], N[1], N[2], cfg)
}
