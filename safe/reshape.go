package safe

func Reshape3DFloat32(array []float32, Nx, Ny, Nz int) [][][]float32 {
	sliced := make([][][]float32, Nx)
	for i := range sliced {
		sliced[i] = make([][]float32, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
		}
	}
	return sliced
}

func Reshape3DComplex64(array []complex64, Nx, Ny, Nz int) [][][]complex64 {
	sliced := make([][][]complex64, Nx)
	for i := range sliced {
		sliced[i] = make([][]complex64, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
		}
	}
	return sliced
}
