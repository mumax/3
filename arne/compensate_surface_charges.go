package arne

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"math"
)

// For a nanowire magnetized like this:
// 	-> <-
// calculate, very approximately, a B field needed to compensate
// for the surface charges on the left and right edges.
// Adding (this field * Bsat) to B_effective will mimic an infinitely long wire.
func CompensateSurfaceCharges(m *nimble.Mesh) [3][][][]float32 {
	core.Log("calculating field to compensate nanowire surface charge")
	H := core.MakeVectors(m.Size())
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
						h1 := Hfield(q, source1, dst)
						h2 := Hfield(q, source2, dst)
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
func Hfield(charge float64, source, dest [3]float64) [3]float64 {
	var R [3]float64
	for i := range R {
		R[i] = dest[i] - source[i]
	}
	r := math.Sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2])
	r3pi4 := 4 * math.Pi * r * r * r
	var h [3]float64
	for i := range h {
		h[i] = R[i] * charge / r3pi4
	}
	return h
}
