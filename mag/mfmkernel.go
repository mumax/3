package mag

import (
	"fmt"
	d "github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

func MFMKernel(mesh *d.Mesh, lift, tipsize float64) (kernel [3]*d.Slice) {

	const TipCharge = 1 // tip charge
	const Δ = 1e-9      // tip oscillation
	util.AssertMsg(lift > 0, "MFM tip crashed into sample, please lift the new one higher")

	{ // Kernel mesh is 2x larger than input, instead in case of PBC
		pbc := mesh.PBC()
		util.AssertMsg(pbc == [3]int{0, 0, 0}, "no PBC for MFM")
		sz := padSize(mesh.Size(), pbc)
		cs := mesh.CellSize()
		mesh = d.NewMesh(sz[X], sz[Y], sz[Z], cs[X], cs[Y], cs[Z], pbc[:]...)
	}

	// Shorthand
	size := mesh.Size()
	cellsize := mesh.CellSize()
	volume := cellsize[X] * cellsize[Y] * cellsize[Z]
	fmt.Print("calculating MFM kernel")

	// Sanity check
	{
		util.Assert(size[Z] >= 1 && size[Y] >= 2 && size[X] >= 2)
		util.Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
		util.AssertMsg(size[X]%2 == 0 && size[Y]%2 == 0, "Even kernel size needed")
		if size[Z] > 1 {
			util.AssertMsg(size[Z]%2 == 0, "Even kernel size needed")
		}
	}

	// Allocate only upper diagonal part. The rest is symmetric due to reciprocity.
	var K [3][][][]float32
	for i := 0; i < 3; i++ {
		kernel[i] = d.NewSlice(1, mesh)
		K[i] = kernel[i].Scalars()
	}

	// Field (destination) loop ranges
	var x1, x2 = -(size[X] - 1) / 2, (size[X] - 1) / 2
	var y1, y2 = -(size[Y] - 1) / 2, (size[Y] - 1) / 2
	var z1, z2 = -(size[Z] - 1) / 2, (size[Z] - 1) / 2
	// support for 2D simulations (thickness 1)
	if size[Z] == 1 {
		z2 = 0
	}

	for s := 0; s < 3; s++ { // source index Ksxyz
		fmt.Print(".")
		m := d.Vector{0, 0, 0}
		m[s] = 1

		for iz := z1; iz <= z2; iz++ {
			zw := wrap(iz, size[Z])
			z := float64(iz) * cellsize[Z]

			for iy := y1; iy <= y2; iy++ {
				yw := wrap(iy, size[Y])
				y := float64(iy) * cellsize[Y]

				for ix := x1; ix <= x2; ix++ {
					x := float64(ix) * cellsize[X]
					xw := wrap(ix, size[X])

					R1 := d.Vector{-x, -y, z - lift}
					r1 := R1.Len()
					B1 := R1.Mul(TipCharge / (4 * math.Pi * r1 * r1 * r1))
					// add 2nd pole here
					E1 := B1.Dot(m) * volume

					R2 := d.Vector{-x, -y, z - (lift + Δ)}
					r2 := R2.Len()
					B2 := R2.Mul(TipCharge / (4 * math.Pi * r2 * r2 * r2))
					// add 2nd pole here
					E2 := B2.Dot(m) * volume

					Fz_tip := (E2 - E1) / Δ

					K[s][zw][yw][xw] = float32(Fz_tip)
					K[s][zw][yw][xw] = 0
				}
			}
		}
	}

	K[X][0][0][0] = 1
	K[Y][0][0][0] = 1
	K[Z][0][0][0] = 1

	fmt.Println()
	d.WriteFile("mfmkx.dump", kernel[X], d.Meta{})
	d.WriteFile("mfmky.dump", kernel[Y], d.Meta{})
	d.WriteFile("mfmkz.dump", kernel[Z], d.Meta{})
	return kernel
}
