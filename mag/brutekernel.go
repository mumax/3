package mag

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"math"
)

// Calculates the magnetostatic kernel by brute-force integration
// of magnetic charges over the faces. Fields are evaluated at the
// cell center (not averaged).
// Mesh should NOT yet be zero-padded.
func BruteKernel(mesh *nimble.Mesh, accuracy float64) [3][3][][][]float32 {
	pbc := mesh.PBC()
	sz := padSize(mesh.Size(), pbc)
	cs := mesh.CellSize()
	mesh = nimble.NewMesh(sz[0], sz[1], sz[2], cs[0], cs[1], cs[2], pbc[:]...)

	size := mesh.Size()
	cellsize := mesh.CellSize()
	periodic := mesh.PBC()
	core.Log("Calculating demag kernel:", "accuracy:", accuracy, ", mesh:", mesh)

	core.Assert(size[0] > 0 && size[1] > 1 && size[2] > 1)
	core.Assert(cellsize[0] > 0 && cellsize[1] > 0 && cellsize[2] > 0)
	core.Assert(periodic[0] >= 0 && periodic[1] >= 0 && periodic[2] >= 0)
	core.Assert(accuracy > 0)
	// TODO: in case of PBC, this will not be met:
	core.Assert(size[1]%2 == 0 && size[2]%2 == 0)
	if size[0] > 1 {
		core.Assert(size[0]%2 == 0)
	}

	var array [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			array[i][j] = core.MakeFloats(size)
		}
	}

	B := [3]float64{0, 0, 0}
	R := [3]float64{0, 0, 0}

	x1 := -(size[X] - 1) / 2
	x2 := size[X]/2 - 1
	// support for 2D simulations (thickness 1)
	if size[X] == 1 && periodic[X] == 0 {
		x2 = 0
	}

	y1 := -(size[Y] - 1) / 2
	y2 := size[Y]/2 - 1

	z1 := -(size[Z] - 1) / 2
	z2 := size[Z]/2 - 1

	x1 *= (periodic[X] + 1)
	x2 *= (periodic[X] + 1)
	y1 *= (periodic[Y] + 1)
	y2 *= (periodic[Y] + 1)
	z1 *= (periodic[Z] + 1)
	z2 *= (periodic[Z] + 1)

	R2 := [3]float64{0, 0, 0}
	pole := [3]float64{0, 0, 0} // position of point charge on the surface

	for s := 0; s < 3; s++ { // source index Ksdxyz
		for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped.
			xw := Wrap(x, size[X])
			R[X] = float64(x) * cellsize[X]
			for y := y1; y <= y2; y++ {
				yw := Wrap(y, size[Y])
				R[Y] = float64(y) * cellsize[Y]
				for z := z1; z <= z2; z++ {
					zw := Wrap(z, size[Z])
					R[Z] = float64(z) * cellsize[Z]

					u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions

					// choose number of integration points depending on how far we are from source.
					r := math.Sqrt(R[X]*R[X] + R[Y]*R[Y] + R[Z]*R[Z])
					nv := int(accuracy*cellsize[v]/(0.5*cellsize[u]+r)) + 1
					nw := int(accuracy*cellsize[w]/(0.5*cellsize[u]+r)) + 1
					scale := 1 / float64(nv*nw)
					surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
					charge := surface * scale

					pu1 := cellsize[u] / 2. // positive pole
					pu2 := -pu1             // negative pole

					B[X], B[Y], B[Z] = 0, 0, 0 // accumulates during surface integral
					for i := 0; i < nv; i++ {
						pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*nv) + float64(i)*(cellsize[v]/float64(nv))
						pole[v] = pv
						for j := 0; j < nw; j++ {
							pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*nw) + float64(j)*(cellsize[w]/float64(nw))
							pole[w] = pw

							pole[u] = pu1
							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]
							r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
							qr := charge / (4 * math.Pi * r * r * r)
							B[X] += R2[X] * qr
							B[Y] += R2[Y] * qr
							B[Z] += R2[Z] * qr

							pole[u] = pu2
							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]
							r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
							qr = -charge / (4 * math.Pi * r * r * r)
							B[X] += R2[X] * qr
							B[Y] += R2[Y] * qr
							B[Z] += R2[Z] * qr
						}
					}

					for d := s; d < 3; d++ { // destination index Ksdxyz
						// TODO: for PBC, need to add here
						array[s][d][xw][yw][zw] = float32(B[d])
					}
				}
			}
		}
	}
	// for 2D these elements are zero:
	if size[0] == 1 {
		array[0][1] = nil
		array[0][2] = nil
	}
	// make result symmetric for tools that expect it so.
	array[1][0] = array[0][1]
	array[2][0] = array[0][2]
	array[2][1] = array[1][2]
	return array
}

const (
	X = 0
	Y = 1
	Z = 2
)

// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func Wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}

// Returns the size after zero-padding,
// taking into account periodic boundary conditions.
func padSize(size, periodic [3]int) [3]int {
	for i := range size {
		if periodic[i] == 0 && size[i] > 1 {
			size[i] *= 2
		}
	}
	return size
}
