package engine

import "math"

func init() {
	world.Func("Ellipsoid", Ellipsoid)
	world.Func("Cylinder", Cylinder)
}

// geometrical shape for setting sample geometry
type Shape func(x, y, z float64) bool

// Primitives:

// Ellipsoid with given diameters
func Ellipsoid(diamx, diamy, diamz float64) Shape {
	return func(x, y, z float64) bool {
		return sqr64(x/diamx)+sqr64(y/diamy)+sqr64(z/diamz) <= 0.25
	}
}

// Elliptic cylinder along z, with given diameters along x and y.
func Cylinder(diamx, diamy float64) Shape {
	return Ellipsoid(diamx, diamy, inf)
}

// Rectangular slab with given sides.
func Rect(sidex, sidey, sidez float64) Shape {
	return func(x, y, z float64) bool {
		rx, ry, rz := sidex/2, sidey/2, sidez/2
		return x < rx && x > -rx && y < ry && y > -ry && z < rz && z > -rz
	}
}

// Part of space with x > 0.
func HalfSpace() Shape {
	return func(x, y, z float64) bool {
		return x > 0
	}
}

// Transforms:

// Transl returns a translated copy of the shape.
func Transl(s Shape, dx, dy, dz float64) Shape {
	return func(x, y, z float64) bool {
		return s(x-dx, y-dy, z-dz)
	}
}

// CSG:

// utils

func sqr64(x float64) float64 { return x * x }

var inf = math.Inf(1)
