package engine

import (
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
	"math"
)

func init() {
	DeclFunc("ext_rmSurfaceCharge", RemoveLRSurfaceCharge, "Compensate magnetic charges on the left and right sides of an in-plane magnetized wire. Arguments: region, mx on left and right side, resp.")
}

// For a nanowire magnetized in-plane, with mx = mxLeft on the left end and
// mx = mxRight on the right end (both -1 or +1), add a B field needed to compensate
// for the surface charges on the left and right edges.
// This will mimic an infinitely long wire.
func RemoveLRSurfaceCharge(region int, mxLeft, mxRight float64) {
	SetBusy(true)
	defer SetBusy(false)
	util.Argument(mxLeft == 1 || mxLeft == -1)
	util.Argument(mxRight == 1 || mxRight == -1)
	bsat := Bsat.GetRegion(region)[0]
	util.AssertMsg(bsat != 0, "RemoveSurfaceCharges: Msat is zero in region "+fmt.Sprint(region))
	B_ext.Add(compensateLRSurfaceCharges(Mesh(), mxLeft, mxRight, bsat), nil)
}

// Returns the saturation magnetization in Tesla.
// Cannot be set. Set Msat and bsat() will automatically be updated.
func bSat() float64 {
	util.AssertMsg(Msat.IsUniform(), "Remove surface charge: Msat must be uniform")
	return mag.Mu0 * Msat.GetRegion(0)
}

func compensateLRSurfaceCharges(m *data.Mesh, mxLeft, mxRight float64, bsat float64) *data.Slice {
	h := data.NewSlice(3, m.Size())
	H := h.Vectors()
	world := m.WorldSize()
	cell := m.CellSize()
	size := m.Size()
	q := cell[Z] * cell[Y]
	q1 := q * mxLeft
	q2 := q * (-mxRight)

	prog, maxProg := 0, (size[Z]+1)*(size[Y]+1)

	// surface loop (source)
	for I := 0; I < size[Z]; I++ {
		for J := 0; J < size[Y]; J++ {
			prog++
			util.Progress(prog, maxProg, "removing surface charges")

			y := (float64(J) + 0.5) * cell[Y]
			z := (float64(I) + 0.5) * cell[Z]
			source1 := [3]float64{0, y, z}        // left surface source
			source2 := [3]float64{world[X], y, z} // right surface source

			// volume loop (destination)
			for iz := range H[0] {
				for iy := range H[0][iz] {
					for ix := range H[0][iz][iy] {

						dst := [3]float64{ // destination coordinate
							(float64(ix) + 0.5) * cell[X],
							(float64(iy) + 0.5) * cell[Y],
							(float64(iz) + 0.5) * cell[Z]}

						h1 := hfield(q1, source1, dst)
						h2 := hfield(q2, source2, dst)

						// add this surface charges' field to grand total
						for c := 0; c < 3; c++ {
							H[c][iz][iy][ix] += float32(h1[c] + h2[c])
						}
					}
				}
			}
		}
	}
	return h
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
