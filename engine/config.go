package engine

// Utilities for setting magnetic configurations.

import (
	"math"
)

func init() {
	World.Func("uniform", Uniform, "Uniform magnetization in given direction")
	World.Func("vortex", Vortex, "Vortex magnetization with given core circulation and polarization")
	World.Func("twodomain", TwoDomain, "Twodomain magnetization with with given magnetization in left domain, wall, and right domain")
	World.Func("vortexwall", VortexWall, "Vortex wall magnetization with given mx in left and right domain and core circulation and polarization")
}

// Magnetic configuration returns m vector for position (x,y,z)
type Config func(x, y, z float64) [3]float64

// Returns a uniform magnetization state. E.g.:
// 	M = Uniform(1, 0, 0)) // saturated along X
func Uniform(mx, my, mz float64) Config {
	return func(x, y, z float64) [3]float64 {
		return [3]float64{mx, my, mz}
	}
}

// Make a vortex magnetization with given circulation and core polarization (+1 or -1).
// The core is smoothed over a few exchange lengths and should easily relax to its ground state.
func Vortex(circ, pol int) Config {
	diam2 := 2 * sqr64(Mesh().CellSize()[2])
	return func(x, y, z float64) [3]float64 {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mx := -y * float64(circ) / r
		my := x * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
		return [3]float64{mx, my, mz}
	}
}

// Make a vortex wall configuration.
func VortexWall(mleft, mright float64, circ, pol int) Config {
	h := Mesh().WorldSize()[1]
	v := Vortex(circ, pol)
	return func(x, y, z float64) [3]float64 {
		if x < -h/2 {
			return [3]float64{mleft, 0, 0}
		}
		if x > h/2 {
			return [3]float64{mright, 0, 0}
		}
		return v(x, y, z)
	}
}

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

// Transl returns a translated copy of configuration c. E.g.:
// 	M = Vortex(1, 1).Transl(100e-9, 0, 0)  // vortex with center at x=100nm
func (c Config) Transl(dx, dy, dz float64) Config {
	return func(x, y, z float64) [3]float64 {
		return c(x-dx, y-dy, z-dz)
	}
}

// Scale returns a scaled copy of configuration c.
func (c Config) Scale(sx, sy, sz float64) Config {
	return func(x, y, z float64) [3]float64 {
		return c(x/sx, y/sy, z/sz)
	}
}

// Rotates the configuration around the Z-axis, over θ radians.
func (c Config) RotZ(θ float64) Config {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) [3]float64 {
		x_ := x*cos + y*sin
		y_ := -x*sin + y*cos
		m := c(x_, y_, z)
		mx_ := m[0]*cos - m[1]*sin
		my_ := m[0]*sin + m[1]*cos
		return [3]float64{mx_, my_, m[2]}
	}
}
