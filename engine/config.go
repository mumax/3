package engine

// Utilities for setting magnetic configurations.
// TODO: use [3][][][]float32, hide data.Slice API.
// Requires careful ZYX translation.
// TODO: already normalize them.

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

func init() {
	parser.AddFunc("uniform", Uniform)
	parser.AddFunc("vortex", Vortex)
	parser.AddFunc("twodomain", TwoDomain)
	parser.AddFunc("vortexwall", VortexWall)
}

// Make a vortex magnetization with given circulation and core polarization (+1 or -1). E.g.:
// 	M.Set(Vortex(1, 1)) // counterclockwise, core up
func Vortex(circ, pol int) *data.Slice {
	util.Argument(circ == 1 || circ == -1)
	util.Argument(pol == 1 || pol == -1)

	m := data.NewSlice(3, Mesh())
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

func VortexWall(mleft, mright float64, circ, pol int) *data.Slice {
	m := data.NewSlice(3, Mesh())
	nx, ny, nz := Nx(), Ny(), Nz()
	SetRegion(m, 0, 0, 0, nx/2, ny, nz, Uniform(mleft, 0, 0))           // left half
	SetRegion(m, nx/2, 0, 0, nx, ny, nz, Uniform(mright, 0, 0))         // right half
	SetRegion(m, nx/2-ny/2, 0, 0, nx/2+ny/2, ny, nz, Vortex(circ, pol)) // center
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
	m := data.NewSlice(3, Mesh())
	Nz := Mesh().Size()[0]
	Ny := Mesh().Size()[1]
	Nx := Mesh().Size()[2]
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
	m := data.NewSlice(3, Mesh())
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
func SetRegion(dst *data.Slice, x1, y1, z1, x2, y2, z2 int, config *data.Slice) {
	v := dst.Vectors()
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
}

func vec(mx, my, mz float64) [3]float32 {
	return [3]float32{float32(mz), float32(my), float32(mx)}
}
