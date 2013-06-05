package engine

// Utilities for setting magnetic configurations.

import (
	"math"
)

func init() {
	world.Func("uniform", Uniform)
	world.Func("vortex", Vortex)
	//	world.Func("twodomain", TwoDomain)
	//	world.Func("vortexwall", VortexWall)
}

// magnetic configuration
type Config func(x, y, z float64) [3]float64

// Returns a uniform magnetization state. E.g.:
// 	M.Set(Uniform(1, 0, 0)) // saturated along X
func Uniform(mx, my, mz float64) Config {
	return func(x, y, z float64) [3]float64 {
		return [3]float64{mx, my, mz}
	}
}

// Make a vortex magnetization with given circulation and core polarization (+1 or -1).
func Vortex(circ, pol int) Config {
	diam2 := 2 * (Aex() / (0.5 * Mu0 * Msat() * Msat())) // inverse core diam squared (roughly)
	return func(x, y, z float64) [3]float64 {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mx := -y * float64(circ) / r
		my := x * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
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
