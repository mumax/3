// TODO: mv
package xc

import ()

func GPlot(out [3][][][]float32, file string) {
	//	var vec [3][][][]float32
	//	for c := 0; c < 3; c++ {
	//		vec[c] = safe.Reshape3DFloat32(out[c], size[0], size[1], size[2])
	//	}
	//	f, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	//	defer f.Close()
	//	core.PanicErr(err)
	//	for i := range vec[0] {
	//		for j := range vec[0][i] {
	//			for k := range vec[0][i][j] {
	//				x, y, z := vec[0][i][j][k], vec[1][i][j][k], vec[2][i][j][k]
	//				norm := fmath.Sqrt(x*x + y*y + z*z)
	//				if norm == 0 {
	//					norm = 1
	//				}
	//				fmt.Fprintln(f, i, j, k, "\t", x/norm, y/norm, z/norm)
	//			}
	//		}
	//	}
}
