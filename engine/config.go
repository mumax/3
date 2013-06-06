package engine

// Utilities for setting magnetic configurations.

import (
	"math"
)

func init() {
	world.Func("uniform", Uniform)
	world.Func("vortex", Vortex)
	world.Func("twodomain", TwoDomain)
	//world.Func("vortexwall", VortexWall)
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
// The core is smoothed over a few exchange lengths and should easily relax to its ground state.
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

// Make a 2-domain configuration with domain wall.
// (mx1, my1, mz1) and (mx2, my2, mz2) are the magnetizations in the left and right domain, respectively.
// (mxwall, mywall, mzwall) is the magnetization in the wall. The wall is smoothed over a few cells so it will
// easily relax to its ground state.
// E.g.:
// 	TwoDomain(1,0,0,  0,1,0,  -1,0,0) // head-to-head domains with transverse (Néel) wall
// 	TwoDomain(1,0,0,  0,0,1,  -1,0,0) // head-to-head domains with perpendicular (Bloch) wall
// 	TwoDomain(0,0,1,  1,0,0,   0,0,-1)// up-down domains with Bloch wall
func TwoDomain(mx1, my1, mz1, mxwall, mywall, mzwall, mx2, my2, mz2 float64) Config {
	ww := 2 * Mesh().CellSize()[2] // wall width in cells
	return func(x, y, z float64) [3]float64 {
		var m [3]float64
		if x < 0 {
			m = [3]float64{mx1, my1, mz1}
		} else {
			m = [3]float64{mx2, my2, mz2}
		}
		gauss := math.Exp(-sqr64(x / ww))
		m[0] = (1-gauss)*m[0] + gauss*mxwall
		m[1] = (1-gauss)*m[1] + gauss*mywall
		m[2] = (1-gauss)*m[2] + gauss*mzwall
		return m
	}
}
