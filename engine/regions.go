package engine

import(
	"code.google.com/p/mx3/util"
	"code.google.com/p/mx3/cuda"
)

type Regions struct{
	gpu *cuda.Bytes
	map_ [][][]byte
	contiguous []byte
}

func (r*Regions)init(){
	gpu := cuda.NewBytes(Mesh())
	list := 
}

// Re-interpret a contiguous array as a multi-dimensional array of given size.
func resizeBytes(array []byte, size [3]int) [][][]byte {
	Nx, Ny, Nz := size[0], size[1], size[2]
	util.Argument(Nx*Ny*Nz == len(array) )
	sliced := make([][][]byte, Nx)
	for i := range sliced {
		sliced[i] = make([][]byte, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
		}
	}
	return sliced
}
