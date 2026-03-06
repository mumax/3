package data

// Cut-out a piece between given bounds (incl, excl)
func Crop(in *Slice, x1, x2, y1, y2, z1, z2 int) *Slice {
	Nx := x2 - x1
	Ny := y2 - y1
	Nz := z2 - z1

	size := [3]int{Nx, Ny, Nz}
	ncomp := in.NComp()

	out := NewSlice(ncomp, size)

	a := in.Tensors()
	b := out.Tensors()

	for c := range ncomp {
		for z := range Nz {
			for y := range Ny {
				for x := range Nx {
					b[c][z][y][x] = a[c][z+z1][y+y1][x+x1]
				}
			}
		}
	}

	return out
}
