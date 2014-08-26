package engine

// Utilities for setting magnetic configurations.

import (
	"github.com/mumax/3/data"
	"math"
	"math/rand"
)

func init() {
	DeclFunc("Uniform", Uniform, "Uniform magnetization in given direction")
	DeclFunc("Vortex", Vortex, "Vortex magnetization with given circulation and core polarization")
	DeclFunc("Antivortex", AntiVortex, "Antivortex magnetization with given circulation and core polarization")
	DeclFunc("NeelSkyrmion", NeelSkyrmion, "Néél skyrmion magnetization with given charge and core polarization")
	DeclFunc("BlochSkyrmion", BlochSkyrmion, "Bloch skyrmion magnetization with given chirality and core polarization")
	DeclFunc("TwoDomain", TwoDomain, "Twodomain magnetization with with given magnetization in left domain, wall, and right domain")
	DeclFunc("VortexWall", VortexWall, "Vortex wall magnetization with given mx in left and right domain and core circulation and polarization")
	DeclFunc("RandomMag", RandomMag, "Random magnetization")
	DeclFunc("RandomMagSeed", RandomMagSeed, "Random magnetization with given seed")
}

// Magnetic configuration returns m vector for position (x,y,z)
type Config func(x, y, z float64) data.Vector

func RandomMag() Config {
	return RandomMagSeed(0)
}

func RandomMagSeed(seed int) Config {
	rng := rand.New(rand.NewSource(int64(seed)))
	return func(x, y, z float64) data.Vector {
		return randomDir(rng)
	}
}

// generate anisotropic random unit vector
func randomDir(rng *rand.Rand) data.Vector {
	theta := 2 * rng.Float64() * math.Pi
	z := 2 * (rng.Float64() - 0.5)
	b := math.Sqrt(1 - z*z)
	x := b * math.Cos(theta)
	y := b * math.Sin(theta)
	return data.Vector{x, y, z}
}

// Returns a uniform magnetization state. E.g.:
// 	M = Uniform(1, 0, 0)) // saturated along X
func Uniform(mx, my, mz float64) Config {
	return func(x, y, z float64) data.Vector {
		return data.Vector{mx, my, mz}
	}
}

// Make a vortex magnetization with given circulation and core polarization (+1 or -1).
// The core is smoothed over a few exchange lengths and should easily relax to its ground state.
func Vortex(circ, pol int) Config {
	diam2 := 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mx := -y * float64(circ) / r
		my := x * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func NeelSkyrmion(charge, pol int) Config {
	w := 8 * Mesh().CellSize()[X]
	w2 := w * w
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mz := 2 * float64(pol) * (math.Exp(-r2/w2) - 0.5)
		mx := (x * float64(charge) / r) * (1 - math.Abs(mz))
		my := (y * float64(charge) / r) * (1 - math.Abs(mz))
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func BlochSkyrmion(charge, pol int) Config {
	w := 8 * Mesh().CellSize()[X]
	w2 := w * w
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mz := 2 * float64(pol) * (math.Exp(-r2/w2) - 0.5)
		mx := (-y * float64(charge) / r) * (1 - math.Abs(mz))
		my := (x * float64(charge) / r) * (1 - math.Abs(mz))
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func AntiVortex(circ, pol int) Config {
	diam2 := 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mx := -x * float64(circ) / r
		my := y * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

// Make a vortex wall configuration.
func VortexWall(mleft, mright float64, circ, pol int) Config {
	h := Mesh().WorldSize()[Y]
	v := Vortex(circ, pol)
	return func(x, y, z float64) data.Vector {
		if x < -h/2 {
			return data.Vector{mleft, 0, 0}
		}
		if x > h/2 {
			return data.Vector{mright, 0, 0}
		}
		return v(x, y, z)
	}
}

func noNaN(v data.Vector, pol int) data.Vector {
	if math.IsNaN(v[X]) || math.IsNaN(v[Y]) || math.IsNaN(v[Z]) {
		return data.Vector{0, 0, float64(pol)}
	} else {
		return v
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
	ww := 2 * Mesh().CellSize()[X] // wall width in cells
	return func(x, y, z float64) data.Vector {
		var m data.Vector
		if x < 0 {
			m = data.Vector{mx1, my1, mz1}
		} else {
			m = data.Vector{mx2, my2, mz2}
		}
		gauss := math.Exp(-sqr64(x / ww))
		m[X] = (1-gauss)*m[X] + gauss*mxwall
		m[Y] = (1-gauss)*m[Y] + gauss*mywall
		m[Z] = (1-gauss)*m[Z] + gauss*mzwall
		return m
	}
}

// Transl returns a translated copy of configuration c. E.g.:
// 	M = Vortex(1, 1).Transl(100e-9, 0, 0)  // vortex with center at x=100nm
func (c Config) Transl(dx, dy, dz float64) Config {
	return func(x, y, z float64) data.Vector {
		return c(x-dx, y-dy, z-dz)
	}
}

// Scale returns a scaled copy of configuration c.
func (c Config) Scale(sx, sy, sz float64) Config {
	return func(x, y, z float64) data.Vector {
		return c(x/sx, y/sy, z/sz)
	}
}

// Rotates the configuration around the Z-axis, over θ radians.
func (c Config) RotZ(θ float64) Config {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) data.Vector {
		x_ := x*cos + y*sin
		y_ := -x*sin + y*cos
		m := c(x_, y_, z)
		mx_ := m[X]*cos - m[Y]*sin
		my_ := m[X]*sin + m[Y]*cos
		return data.Vector{mx_, my_, m[Z]}
	}
}

func (c Config) Add(weight float64, other Config) Config {
	return func(x, y, z float64) data.Vector {
		return c(x, y, z).MAdd(weight, other(x, y, z))
	}
}

// Infinitely repeats the shape with given period in x, y, z.
// A period of 0 or infinity means no repetition.
//func (c Config) Repeat(periodX, periodY, periodZ float64) Config {
//	return func(x, y, z float64) data.Vector {
//		return c(fmod(x, periodX), fmod(y, periodY), fmod(z, periodZ))
//	}
//}
