package mag

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"log"
	"math"
)

// Calculates the magnetostatic kernel by brute-force integration
// of magnetic charges over the faces and averages over cell volumes.
// Mesh should NOT yet be zero-padded.
func BruteKernel(mesh *data.Mesh, accuracy float64) (kernel [3][3]*data.Slice) {

	{ // Kernel mesh is 2x larger than input, instead in case of PBC
		pbc := mesh.PBC()
		util.Argument(pbc == [3]int{0, 0, 0}) // PBC not supported yet
		sz := padSize(mesh.Size(), pbc)
		cs := mesh.CellSize()
		mesh = data.NewMesh(sz[0], sz[1], sz[2], cs[0], cs[1], cs[2], pbc[:]...)
	}

	// Shorthand
	size := mesh.Size()
	cellsize := mesh.CellSize()
	periodic := mesh.PBC()
	log.Println("calculating demag kernel:", "accuracy:", accuracy, ", size:", size[0], "x", size[1], "x", size[2])

	// Sanity check
	{
		util.Assert(size[0] > 0 && size[1] > 1 && size[2] > 1)
		util.Assert(cellsize[0] > 0 && cellsize[1] > 0 && cellsize[2] > 0)
		util.Assert(periodic[0] >= 0 && periodic[1] >= 0 && periodic[2] >= 0)
		util.Assert(accuracy > 0)
		// TODO: in case of PBC, this will not be met:
		util.Assert(size[1]%2 == 0 && size[2]%2 == 0)
		if size[0] > 1 {
			util.Assert(size[0]%2 == 0)
		}
	}

	// Allocate only upper diagonal part. The rest is symmetric due to reciprocity.
	var array [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			kernel[i][j] = data.NewSlice(1, mesh)
			array[i][j] = kernel[i][j].Scalars()
		}
	}

	// Field (destination) loop ranges
	x1, x2 := -(size[X]-1)/2, size[X]/2-1
	y1, y2 := -(size[Y]-1)/2, size[Y]/2-1
	z1, z2 := -(size[Z]-1)/2, size[Z]/2-1
	// support for 2D simulations (thickness 1)
	if size[X] == 1 && periodic[X] == 0 {
		x2 = 0
	}
	{ // Repeat for PBC:
		x1 *= (periodic[X] + 1)
		x2 *= (periodic[X] + 1)
		y1 *= (periodic[Y] + 1)
		y2 *= (periodic[Y] + 1)
		z1 *= (periodic[Z] + 1)
		z2 *= (periodic[Z] + 1)
	}

	// smallest cell dimension is our typical length scale
	L := cellsize[X]
	if cellsize[Y] < L {
		L = cellsize[Y]
	}
	if cellsize[Z] < L {
		L = cellsize[Z]
	}

	// Start brute integration
	// 9 nested loops, does that stress you out?
	// Fortunately, the 5 inner ones usually loop over just one element.
	// It might be nice to get rid of that branching though.
	var (
		R, R2  [3]float64 // field and source cell center positions
		pole   [3]float64 // position of point charge on the surface
		points int        // counts used integration points
	)
	for s := 0; s < 3; s++ { // source index Ksdxyz
		u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions

		for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped.
			xw := wrap(x, size[X])
			R[X] = float64(x) * cellsize[X]

			for y := y1; y <= y2; y++ {
				yw := wrap(y, size[Y])
				R[Y] = float64(y) * cellsize[Y]

				for z := z1; z <= z2; z++ {
					zw := wrap(z, size[Z])
					R[Z] = float64(z) * cellsize[Z]

					// choose number of integration points depending on how far we are from source.
					dx, dy, dz := delta(x)*cellsize[X], delta(y)*cellsize[Y], delta(z)*cellsize[Z]
					d := math.Sqrt(dx*dx + dy*dy + dz*dz)
					if d == 0 {
						d = L
					}
					maxSize := d / accuracy // maximum acceptable integration size
					nv := int(math.Max(cellsize[v]/maxSize, 1) + 0.5)
					nw := int(math.Max(cellsize[w]/maxSize, 1) + 0.5)
					nx := int(math.Max(cellsize[X]/maxSize, 1) + 0.5)
					ny := int(math.Max(cellsize[Y]/maxSize, 1) + 0.5)
					nz := int(math.Max(cellsize[Z]/maxSize, 1) + 0.5)
					// Stagger source and destination grids.
					// Massively improves accuracy. Could play with variations.
					// See note.
					nv *= 2
					nw *= 2

					util.Assert(nv > 0 && nw > 0 && nx > 0 && ny > 0 && nz > 0)

					scale := 1 / float64(nv*nw*nx*ny*nz)
					surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
					charge := surface * scale
					pu1 := cellsize[u] / 2. // positive pole center
					pu2 := -pu1             // negative pole center

					// Do surface integral over source cell, accumulate  in B
					var B [3]float64
					for i := 0; i < nv; i++ {
						pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*nv) + float64(i)*(cellsize[v]/float64(nv))
						pole[v] = pv
						for j := 0; j < nw; j++ {
							pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*nw) + float64(j)*(cellsize[w]/float64(nw))
							pole[w] = pw

							// Do volume integral over destination cell
							for α := 0; α < nx; α++ {
								rx := R[X] - cellsize[X]/2 + cellsize[X]/float64(2*nx) + (cellsize[X]/float64(nx))*float64(α)

								for β := 0; β < ny; β++ {
									ry := R[Y] - cellsize[Y]/2 + cellsize[Y]/float64(2*ny) + (cellsize[Y]/float64(ny))*float64(β)

									for γ := 0; γ < nz; γ++ {
										rz := R[Z] - cellsize[Z]/2 + cellsize[Z]/float64(2*nz) + (cellsize[Z]/float64(nz))*float64(γ)
										points++

										pole[u] = pu1
										R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
										r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
										qr := charge / (4 * math.Pi * r * r * r)
										bx := R2[X] * qr
										by := R2[Y] * qr
										bz := R2[Z] * qr

										pole[u] = pu2
										R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
										r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
										qr = -charge / (4 * math.Pi * r * r * r)
										B[X] += (bx + R2[X]*qr) // addition ordered for accuracy
										B[Y] += (by + R2[Y]*qr)
										B[Z] += (bz + R2[Z]*qr)

									}
								}
							}
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
	log.Println("kernel used", points, "integration points")
	// for 2D these elements are zero:
	if size[0] == 1 {
		kernel[0][1] = nil
		kernel[0][2] = nil
	}
	// make result symmetric for tools that expect it so.
	kernel[1][0] = kernel[0][1]
	kernel[2][0] = kernel[0][2]
	kernel[2][1] = kernel[1][2]
	return kernel
}

const (
	X = 0
	Y = 1
	Z = 2
)

// closest distance between cells, given center distance d.
// if cells touch by just even a corner, the distance is zero.
func delta(d int) float64 {
	if d < 0 {
		d = -d
	}
	if d > 0 {
		d -= 1
	}
	return float64(d)
}

// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func wrap(number, max int) int {
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

// "If brute force doesn't solve your problem,
// you're not using enough of it."

/*
	Note: error for cubic self-kernel for different stagger decisions:

       1 ++--+----+-++---+----+--++---+----+-++---+----+--++---+----+-++--++
         +           +            +           +            +           +   +
         |                                                                 |
         +                 A                                               +
     0.1 ++                           A      A                            ++
         +                                         A    A   A   A          +
         +                       C                                 A  A  A +
    0.01 ++    B               D          E      C                        ++
 e       +                               B      D     E    C               +
 r       |                        F                  B    D   BE  C        |
 r       +                                   F                   D  BE DC B+
 o 0.001 ++                                                               ++
 r       +                                           F                     +
         +                                                 F               +
  0.0001 ++                                                             F +F
         +                                                          F      +
         |                                                      F          |
         +           +            +           +            +           +   +
   1e-05 ++--+----+-++---+----+--++---+----+-++---+----+--++---+----+-++--++
        100         1000        10000       100000       1e+06       1e+07
                                  evaluation points
A: no staggering

B:
	nv = ((nv + 1) / 2) * 2
	nw = ((nw + 1) / 2) * 2
	nx = ((nx+1)/2)*2 - 1
	ny = ((ny+1)/2)*2 - 1
	nz = ((nz+1)/2)*2 - 1

C:
	nv = ((nv + 1) / 2) * 2
	nw = ((nw + 1) / 2) * 2
	nx = ((nx+1)/2)*2 + 1
	ny = ((ny+1)/2)*2 + 1
	nz = ((nz+1)/2)*2 + 1

D:
	nv += 1
	nw += 1

E:
	nx += 1
	ny += 1
	nz += 1

F: best with accuracy 6
	nv *= 2
	nw *= 2
*/
