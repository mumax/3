package data

func Crop(in *Slice, x1, x2, y1, y2, z1, z2 int) *Slice {
	Nx := x2 - x1
	Ny := y2 - y1
	Nz := z2 - z1

	mesh := *in.Mesh()
	mesh.gridSize = [3]int{Nx, Ny, Nz}
	ncomp := in.NComp()

	out := NewSlice(ncomp, &mesh)

	a := in.Tensors()
	b := out.Tensors()

	for c := 0; c < ncomp; c++ {
		for i := 0; i < Nx; i++ {
			for j := 0; j < Ny; j++ {
				for k := 0; k < Nz; k++ {
					b[c][i][j][k] = a[c][i+x1][j+y1][k+z1]
				}
			}
		}
	}

	return out
}
