package engine

// Utilities for setting magnetic configurations.
// TODO: use [3][][][]float32, hide data.Slice API.
// Requires careful ZYX translation.
// TODO: already normalize them.

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Make a vortex magnetization with given circulation and core polarization (+1 or -1). E.g.:
// 	M.Set(Vortex(1, 1)) // counterclockwise, core up
func Vortex(circ, pol int) *data.Slice {
	util.Argument(circ == 1 || circ == -1)
	util.Argument(pol == 1 || pol == -1)

	m := data.NewSlice(3, &mesh)
	v := m.Vectors()
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
	return m
}

// Make a 2-domain configuration with domain wall.
// (mx1, my1, mz1) and (mx2, my2, mz2) are the magnetizations in the left and right domain, respectively.
// (mxwall, mywall, mzwall) is the magnetization in the wall.
// E.g.:
// 	M.Set(TwoDomain(1,0,0,  0,1,0,  -1,0,0)) // head-to-head domains with transverse (Néel) wall
// 	M.Set(TwoDomain(1,0,0,  0,0,1,  -1,0,0)) // head-to-head domains with perpendicular (Bloch) wall
// 	M.Set(TwoDomain(0,0,1,  1,0,0,   0,0,-1))// up-down domains with Bloch wall
func TwoDomain(mx1, my1, mz1, mxwall, mywall, mzwall, mx2, my2, mz2 float64) *data.Slice {
	m1 := vec(mx1, my1, mz1)
	m2 := vec(mx2, my2, mz2)
	mw := vec(mxwall, mywall, mzwall)
	m := data.NewSlice(3, &mesh)
	Nz := mesh.Size()[0]
	Ny := mesh.Size()[1]
	Nx := mesh.Size()[2]
	util.Argument(Nx >= 4)
	v := m.Vectors()
	for c := range mw {
		for i := 0; i < Nz; i++ {
			for j := 0; j < Ny; j++ {
				for k := 0; k < Nx/2; k++ {
					v[c][i][j][k] = m1[c]
				}
				for k := Nx / 2; k < Nx; k++ {
					v[c][i][j][k] = m2[c]
				}
				v[c][i][j][Nx/2-2] += mw[c]
				v[c][i][j][Nx/2-1] = mw[c]
				v[c][i][j][Nx/2] = mw[c]
				v[c][i][j][Nx/2+1] += mw[c]
			}
		}
	}
	return m
}

// Returns a uniform magnetization state. E.g.:
// 	M.Set(Uniform(1, 0, 0)) // saturated along X
func Uniform(mx, my, mz float64) *data.Slice {
	m := data.NewSlice(3, &mesh)
	v := vec(mx, my, mz)
	list := m.Host()
	for c := 0; c < m.NComp(); c++ {
		for i := range list[c] {
			list[c][i] = v[c]
		}
	}
	return m
}

// Only sets the region between cells [x1, y1, z1] and [x2, y2, z2] (excl.) to the given configuration.
// E.g.: to set m to something resembling a vortex wall:
// 	// Nx, Ny, Nz= number of cells
// 	M.SetRegion(0,    0, 0,   Nx/2, Ny, Nz,  Uniform( 1, 0, 0)) // left half
// 	M.SetRegion(Nx/2, 0, 0,   Nx,   Ny, Nz,  Uniform(-1, 0, 0)) // right half
// 	M.SetRegion(Nx/2-Ny/2, 0, 0,   Nx/2+Ny/2,   Ny, Nz,  Vortex(1, 1)) // center
func (M *Magnetization) SetRegion(x1, y1, z1, x2, y2, z2 int, config *data.Slice) {

	m := M.Download()
	v := m.Vectors()
	src := config.Vectors()

	for c := range v {
		for i := z1; i < z2; i++ {
			for j := y1; j < y2; j++ {
				for k := x1; k < x2; k++ {
					v[c][i][j][k] = src[c][i][j][k]
				}
			}
		}
	}

	M.Set(m)
}

func vec(mx, my, mz float64) [3]float32 {
	return [3]float32{float32(mz), float32(my), float32(mx)}
}

// convert mumax's internal ZYX convention to userspace XYZ.
//func convertXYZ(arr [][][][]float32) *host.Array {
//	s := arr.Size3D
//	n := arr.NComp()
//	a := arr.Array
//	transp := host.NewArray(n, []int{s[Z], s[Y], s[X]})
//	t := transp.Array
//	for c := 0; c < n; c++ {
//		c2 := swapIndex(c, n)
//		for i := 0; i < s[X]; i++ {
//			for j := 0; j < s[Y]; j++ {
//				for k := 0; k < s[Z]; k++ {
//					t[c2][k][j][i] = a[c][i][j][k]
//				}
//			}
//		}
//	}
//	return transp
//}
