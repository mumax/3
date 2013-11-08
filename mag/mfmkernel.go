package mag

import (
	"fmt"
	d "github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

func MFMKernel(mesh *d.Mesh, lift, tipsize float64) (kernel [3]*d.Slice) {

	const TipCharge = 1e-18 // tip charge
	const Δ = 1e-9          // tip oscillation
	util.AssertMsg(lift != 0, "MFM tip just crashed into sample, buy new one")

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

					R2 := d.Vector{-x, -y, z - (lift + Δ)}
					r2 := R2.Len()
					B2 := R2.Mul(TipCharge / (4 * math.Pi * r2 * r2 * r2))

					Fz_tip := ((B2.Sub(B1)).Div(Δ))[s] // dot m_s (=1)

					K[s][zw][yw][xw] = float32(Fz_tip)
				}
			}
		}
	}
	fmt.Println(kernel)
	return kernel
}
