package mag

import (
	"fmt"
	d "github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"log"
	"math"
)

func MFMKernel(mesh *d.Mesh, lift, tipsize float64) (kernel [3]*d.Slice) {

	const TipCharge = 1 / Mu0 // tip charge
	const Δ = 1e-9            // tip oscillation
	util.AssertMsg(lift > 0, "MFM tip crashed into sample, please lift the new one higher")

	{ // Kernel mesh is 2x larger than input, instead in case of PBC
		pbc := mesh.PBC()
		sz := padSize(mesh.Size(), pbc)
		cs := mesh.CellSize()
		mesh = d.NewMesh(sz[X], sz[Y], sz[Z], cs[X], cs[Y], cs[Z], pbc[:]...)
	}

	// Shorthand
	size := mesh.Size()
	pbc := mesh.PBC()
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
		kernel[i] = d.NewSlice(1, mesh.Size())
		K[i] = kernel[i].Scalars()
	}

	r1, r2 := kernelRanges(size, pbc)
	log.Println("mfm kernel ranges:", r1, r2)

	for iz := r1[Z]; iz <= r2[Z]; iz++ {
		zw := wrap(iz, size[Z])
		z := float64(iz) * cellsize[Z]

		for iy := r1[Y]; iy <= r2[Y]; iy++ {
			yw := wrap(iy, size[Y])
			y := float64(iy) * cellsize[Y]
			util.Progress((iz-r1[Z])*(r2[Y]-r1[Y])+(iy-r1[Y])+1, (1+r2[Y]-r1[Y])*(1+r2[Z]-r1[Z]))

			for ix := r1[X]; ix <= r2[X]; ix++ {
				x := float64(ix) * cellsize[X]
				xw := wrap(ix, size[X])

				for s := 0; s < 3; s++ { // source index Ksxyz
					m := d.Vector{0, 0, 0}
					m[s] = 1

					R1 := d.Vector{-x, -y, z - lift}
					r1 := R1.Len()
					B1 := R1.Mul(TipCharge / (4 * math.Pi * r1 * r1 * r1))

					R1 = d.Vector{-x, -y, z - (lift + tipsize)}
					r1 = R1.Len()
					B1 = B1.Add(R1.Mul(-TipCharge / (4 * math.Pi * r1 * r1 * r1)))

					E1 := B1.Dot(m) * volume

					R2 := d.Vector{-x, -y, z - (lift + Δ)}
					r2 := R2.Len()
					B2 := R2.Mul(TipCharge / (4 * math.Pi * r2 * r2 * r2))

					R2 = d.Vector{-x, -y, z - (lift + tipsize + Δ)}
					r2 = R2.Len()
					B2 = B2.Add(R2.Mul(-TipCharge / (4 * math.Pi * r2 * r2 * r2)))

					E2 := B2.Dot(m) * volume

					Fz_tip := (E2 - E1) / Δ

					K[s][zw][yw][xw] += float32(Fz_tip) // += needed in case of PBC
				}
			}
		}
	}

	fmt.Println()
	return kernel
}
