package engine

import (
	"code.google.com/p/mx3/data"
	"log"
	"math"
)

// For a nanowire magnetized like this:
// 	-> <-
// calculate, approximately, a B field needed to compensate
// for the surface charges on the left and right edges.
// Adding (this field * Bsat) to B_effective will mimic an infinitely long wire.
func compensateLRSurfaceCharges(m *data.Mesh) [3][][][]float32 {
	log.Println("calculating field to compensate nanowire surface charge")
	h := data.NewSlice(3, m)
	H := h.Vectors()
	world := m.WorldSize()
	cell := m.CellSize()
	size := m.Size()
	q := cell[0] * cell[1]

	for I := 0; I < size[0]; I++ {
		for J := 0; J < size[1]; J++ {

			x := (float64(I) + 0.5) * cell[0]
			y := (float64(J) + 0.5) * cell[1]
			source1 := [3]float64{x, y, 0}
			source2 := [3]float64{x, y, world[2]}
			for i := range H[0] {
				for j := range H[0][i] {
					for k := range H[0][i][j] {
						dst := [3]float64{(float64(i) + 0.5) * cell[0],
							(float64(j) + 0.5) * cell[1],
							(float64(k) + 0.5) * cell[2]}
						h1 := hfield(q, source1, dst)
						h2 := hfield(q, source2, dst)
						for c := 0; c < 3; c++ {
							H[c][i][j][k] += float32(h1[c] + h2[c])
						}
					}
				}
			}
		}
	}
	return H
}

// H field of charge at location source, evaluated in location dest.
func hfield(charge float64, source, dest [3]float64) [3]float64 {
	var R [3]float64
	R[0] = dest[0] - source[0]
	R[1] = dest[1] - source[1]
	R[2] = dest[2] - source[2]
	r := math.Sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2])
	qr3pi4 := charge / ((4 * math.Pi) * r * r * r)
	var h [3]float64
	h[0] = R[0] * qr3pi4
	h[1] = R[1] * qr3pi4
	h[2] = R[2] * qr3pi4
	return h
}
