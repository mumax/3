package mx

// This file implements functions to construct multi-dimensional
// arrays that are backed by a single contiguous list.
// We can thus access the same data as a 1D list or a multi-dimensional
// array. As a bonus, all the data is stored in one contiguous memory
// block, so caching works well. 

import ()

// Allocates a 2D array, as well as the contiguous 1D array backing it.
func Array2D(size0, size1 int) ([]float32, [][]float32) {
	// First make the slice and then the list. When the memory is not fragmented,
	// they are probably allocated in a good order for the cache.
	sliced := make([][]float32, size0)
	list := make([]float32, size0*size1)
	for i := 0; i < size0; i++ {
		sliced[i] = list[i*size1 : (i+1)*size1]
	}
	return list, sliced
}

// Makes a 2D array from a contiguous 1D list
func Slice2D(list []float32, size []int) [][]float32 {
	sliced := make([][]float32, size[0])
	for i := 0; i < size[0]; i++ {
		sliced[i] = list[i*size[1] : (i+1)*size[1]]
	}
	return sliced
}

// Allocates a 3D array, as well as the contiguous 1D array backing it.
func Array3D(size0, size1, size2 int) ([]float32, [][][]float32) {
	sliced := make([][][]float32, size0)
	for i := range sliced {
		sliced[i] = make([][]float32, size1)
	}
	list := make([]float32, size0*size1*size2)
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = list[(i*size1+j)*size2+0 : (i*size1+j)*size2+size2]
		}
	}
	return list, sliced
}

// Makes a 3D array from a contiguous 1D list.
func Slice3D(list []float32, size []int) [][][]float32 {
	sliced := make([][][]float32, size[0])
	for i := range sliced {
		sliced[i] = make([][]float32, size[1])
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = list[(i*size[1]+j)*size[2]+0 : (i*size[1]+j)*size[2]+size[2]]
		}
	}
	return sliced
}

// Allocates a 4D array, as well as the contiguous 1D array backing it.
func Array4D(size0, size1, size2, size3 int) ([]float32, [][][][]float32) {
	sliced := make([][][][]float32, size0)
	for i := range sliced {
		sliced[i] = make([][][]float32, size1)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float32, size2)
		}
	}
	list := make([]float32, size0*size1*size2*size3)
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*size1+j)*size2+k)*size3+0 : ((i*size1+j)*size2+k)*size3+size3]
			}
		}
	}
	return list, sliced
}

// Makes a 4D array from a contiguous 1D list.
func Slice4D(list []float32, size []int) [][][][]float32 {

	sliced := make([][][][]float32, size[0])
	for i := range sliced {
		sliced[i] = make([][][]float32, size[1])
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float32, size[2])
		}
	}

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*size[1]+j)*size[2]+k)*size[3]+0 : ((i*size[1]+j)*size[2]+k)*size[3]+size[3]]
			}
		}
	}
	return sliced
}

// Allocates a 5D array, as well as the contiguous 1D array backing it.
func Array5D(size0, size1, size2, size3, size4 int) ([]float32, [][][][][]float32) {
	sliced := make([][][][][]float32, size0)
	for i := range sliced {
		sliced[i] = make([][][][]float32, size1)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][][]float32, size2)
		}
	}
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = make([][]float32, size3)
			}
		}
	}
	list := make([]float32, size0*size1*size2*size3*size4)
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				for l := range sliced[i][j][k] {
					sliced[i][j][k][l] = list[(((i*size1+j)*size2+k)*size3+l)*size4+0 : (((i*size1+j)*size2+k)*size3+l)*size4+size4]
				}
			}
		}
	}
	return list, sliced
}
