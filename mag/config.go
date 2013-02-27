package mag

// Utilities for setting magnetic configurations.

import (
	"code.google.com/p/mx3/data"
)

// Sets m to a magnetic vortex with given circulation and core polarization (1 or -1).
func SetVortex(m *data.Slice, circ, pol int) {
	mh := host(m)

	v := mh.Vectors()
	cy, cz := len(v[0][0])/2, len(v[0][0][0])/2
	for i := range v[0] {
		for j := range v[0][i] {
			for k := range v[0][0][j] {
				y := j - cy
				x := k - cz
				v[X][i][j][k] = 0
				v[Y][i][j][k] = float32(x * circ)
				v[Z][i][j][k] = float32(-y * circ)
			}
		}
		v[Z][i][cy][cz] = 0.
		v[Y][i][cy][cz] = 0.
		v[X][i][cy][cz] = float32(pol)
	}

	if mh != m {
		data.Copy(m, mh)
	}
}

// return m if already on host, new CPU slice otherwise.
func host(m *data.Slice) *data.Slice {
	if m.CPUAccess() {
		return m
	} //else
	return data.NewSlice(m.NComp(), m.Mesh())
}

//// Returns a function that returns the vector value for all i,j,k.
//func Uniform(x, y, z float32) func(i, j, k int) [3]float32 {
//	v := [3]float32{x, y, z}
//	return func(i, j, k int) [3]float32 {
//		return v
//	}
//}
//
//// Sets value at index i,j,k to f(i,j,k).
//func SetAll(array [3][][][]float32, f func(i, j, k int) [3]float32) {
//	n := core.SizeOf(array[0])
//	i2, j2, k2 := n[0], n[1], n[2]
//	SetRegion(array, 0, 0, 0, i2, j2, k2, f)
//}
//
//// Sets the region between (i1, j1, k1), (i2, j2, k2) to f(i,j,k).
//func SetRegion(array [3][][][]float32, i1, j1, k1, i2, j2, k2 int, f func(i, j, k int) [3]float32) {
//	for i := i1; i < i2; i++ {
//		for j := j1; j < j2; j++ {
//			for k := k1; k < k2; k++ {
//				v := f(i, j, k)
//				for c := range array {
//					array[c][i][j][k] = v[c]
//				}
//			}
//		}
//	}
//}
