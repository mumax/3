package xc

import (
	"github.com/barnex/cuda4/safe"
	"math"
	"nimble-cube/core"
)

// Calculates the magnetostatic kernel
func magKernel(size [3]int, cellsize [3]float64, periodic [3]int, accuracy int) [3][3][]float32 {
	core.Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "accuracy:", accuracy, "periodic:", periodic)

	N := prod(size)

	var kern [3][3][]float32
	var array [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			kern[i][j] = make([]float32, N)
			array[i][j] = safe.Reshape3DFloat32(kern[i][j], size[0], size[1], size[2])
		}
	}

	//array[0][0][0][0][0] = 1

	//return kern

	B := [3]float64{0, 0, 0} //NewVector()
	R := [3]float64{0, 0, 0} //NewVector()

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
			for y := y1; y <= y2; y++ {
				yw := Wrap(y, size[Y])
				for z := z1; z <= z2; z++ {
					zw := Wrap(z, size[Z])
					R[X] = float64(x) * cellsize[X]
					R[Y] = float64(y) * cellsize[Y]
					R[Z] = float64(z) * cellsize[Z]

					n := accuracy                  // number of integration points = n^2
					u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions

					R2[X], R2[Y], R2[Z] = 0, 0, 0
					pole[X], pole[Y], pole[Z] = 0, 0, 0

					surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
					charge := surface

					pu1 := cellsize[u] / 2. // positive pole
					pu2 := -pu1             // negative pole

					B[X], B[Y], B[Z] = 0, 0, 0 // accumulates magnetic field
					for i := 0; i < n; i++ {
						pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*n) + float64(i)*(cellsize[v]/float64(n))
						for j := 0; j < n; j++ {
							pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*n) + float64(j)*(cellsize[w]/float64(n))

							pole[u] = pu1
							pole[v] = pv
							pole[w] = pw

							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]
							r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
							B[X] += R2[X] * charge / (4 * math.Pi * r * r * r)
							B[Y] += R2[Y] * charge / (4 * math.Pi * r * r * r)
							B[Z] += R2[Z] * charge / (4 * math.Pi * r * r * r)

							pole[u] = pu2
							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]
							r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
							B[X] += R2[X] * -charge / (4 * math.Pi * r * r * r)
							B[Y] += R2[Y] * -charge / (4 * math.Pi * r * r * r)
							B[Z] += R2[Z] * -charge / (4 * math.Pi * r * r * r)
						}
					}
					scale := 1 / float64(n*n)
					B[X] *= scale
					B[Y] *= scale
					B[Z] *= scale

					for d := s; d < 3; d++ { // destination index Ksdxyz
						array[s][d][xw][yw][zw] += float32(B[d]) // Add: may have multiple contributions in case of periodicity
					}
				}
			}
		}
	}

	if core.DEBUG {
		for i := 0; i < 3; i++ {
			for j := i; j < 3; j++ {
				core.Debug("kern", i, j, ":", core.Format(array[i][j]))
			}
		}
	}
	return kern
}

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

const (
	X = 0
	Y = 1
	Z = 2
)
