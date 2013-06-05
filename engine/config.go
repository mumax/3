package engine

// Utilities for setting magnetic configurations.

import (
	"code.google.com/p/mx3/data"
	"math"
)

//func init() {
//	world.Func("uniform", Uniform)
//	world.Func("vortex", Vortex)
//	world.Func("twodomain", TwoDomain)
//	world.Func("vortexwall", VortexWall)
//}

type Config func(x, y, z float64) [3]float64

func setConfig(dst *data.Slice, cfg Config, region Shape) {

	if region == nil {
		region = universe
	}

	host := hostBuf(3, dst.Mesh())
	h := host.Vectors()
	n := dst.Mesh().Size()
	c := dst.Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx

				inside := region(x, y, z)
				if inside {
					m := cfg(x, y, z)
					h[0][i][j][k] = float32(m[2])
					h[1][i][j][k] = float32(m[1])
					h[2][i][j][k] = float32(m[0])
				}

			}
		}
	}
}

// Make a vortex magnetization with given circulation and core polarization (+1 or -1). E.g.:
// 	M.Set(Vortex(1, 1)) // counterclockwise, core up
func Vortex(circ, pol int) Config {
	lex2 := Aex() / (0.5 * Mu0 * Msat() * Msat()) // exchange length²
	return func(x, y, z float64) [3]float64 {
		mx := -y * float64(circ)
		my := x * float64(circ)
		r2 := x*x + y*y
		mz := float64(pol) * math.Exp(-r2/lex2)
		return [3]float64{mx, my, mz}
	}
}

//func VortexWall(mleft, mright float64, circ, pol int) *data.Slice {
//	m := data.NewSlice(3, Mesh())
//	nx, ny, nz := Nx(), Ny(), Nz()
//	SetRegion(m, 0, 0, 0, nx/2, ny, nz, Uniform(mleft, 0, 0))           // left half
//	SetRegion(m, nx/2, 0, 0, nx, ny, nz, Uniform(mright, 0, 0))         // right half
//	SetRegion(m, nx/2-ny/2, 0, 0, nx/2+ny/2, ny, nz, Vortex(circ, pol)) // center
//	return m
//}
//
//// Make a 2-domain configuration with domain wall.
//// (mx1, my1, mz1) and (mx2, my2, mz2) are the magnetizations in the left and right domain, respectively.
//// (mxwall, mywall, mzwall) is the magnetization in the wall.
//// E.g.:
//// 	M.Set(TwoDomain(1,0,0,  0,1,0,  -1,0,0)) // head-to-head domains with transverse (Néel) wall
//// 	M.Set(TwoDomain(1,0,0,  0,0,1,  -1,0,0)) // head-to-head domains with perpendicular (Bloch) wall
//// 	M.Set(TwoDomain(0,0,1,  1,0,0,   0,0,-1))// up-down domains with Bloch wall
//func TwoDomain(mx1, my1, mz1, mxwall, mywall, mzwall, mx2, my2, mz2 float64) *data.Slice {
//	m1 := vec(mx1, my1, mz1)
//	m2 := vec(mx2, my2, mz2)
//	mw := vec(mxwall, mywall, mzwall)
//	m := data.NewSlice(3, Mesh())
//	Nz := Mesh().Size()[0]
//	Ny := Mesh().Size()[1]
//	Nx := Mesh().Size()[2]
//	util.Argument(Nx >= 4)
//	v := m.Vectors()
//	for c := range mw {
//		for i := 0; i < Nz; i++ {
//			for j := 0; j < Ny; j++ {
//				for k := 0; k < Nx/2; k++ {
//					v[c][i][j][k] = m1[c]
//				}
//				for k := Nx / 2; k < Nx; k++ {
//					v[c][i][j][k] = m2[c]
//				}
//				v[c][i][j][Nx/2-2] += mw[c]
//				v[c][i][j][Nx/2-1] = mw[c]
//				v[c][i][j][Nx/2] = mw[c]
//				v[c][i][j][Nx/2+1] += mw[c]
//			}
//		}
//	}
//	return m
//}
//
//// Returns a uniform magnetization state. E.g.:
//// 	M.Set(Uniform(1, 0, 0)) // saturated along X
//func Uniform(mx, my, mz float64) *data.Slice {
//	m := data.NewSlice(3, Mesh())
//	v := vec(mx, my, mz)
//	list := m.Host()
//	for c := 0; c < m.NComp(); c++ {
//		for i := range list[c] {
//			list[c][i] = v[c]
//		}
//	}
//	return m
//}
//
//// Only sets the region between cells [x1, y1, z1] and [x2, y2, z2] (excl.) to the given configuration.
//func SetRegion(dst *data.Slice, x1, y1, z1, x2, y2, z2 int, config *data.Slice) {
//	v := dst.Vectors()
//	src := config.Vectors()
//
//	for c := range v {
//		for i := z1; i < z2; i++ {
//			for j := y1; j < y2; j++ {
//				for k := x1; k < x2; k++ {
//					v[c][i][j][k] = src[c][i][j][k]
//				}
//			}
//		}
//	}
//}
//
