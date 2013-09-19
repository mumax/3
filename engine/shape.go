package engine

import (
	"log"
	"math"
)

func init() {
	DeclFunc("Ellipsoid", Ellipsoid, "3D Ellipsoid with axes in meter")
	DeclFunc("Ellipse", Ellipse, "2D Ellipse with axes in meter")
	DeclFunc("Cylinder", Cylinder, "3D Cylinder with diameter and height in meter")
	DeclFunc("Circle", Circle, "2D Circle with diameter in meter")
	DeclFunc("Cuboid", Cuboid, "Cuboid with sides in meter")
	DeclFunc("Rect", Rect, "2D rectangle with size in meter")
	DeclFunc("XRange", XRange, "Part of space between x1 and x2, in meter")
	DeclFunc("YRange", YRange, "Part of space between y1 and y2, in meter")
	DeclFunc("ZRange", ZRange, "Part of space between z1 and z2, in meter")
	DeclFunc("Layers", Layers, "Part of space between cell layer1 (inclusive) and layer2 (exclusive), in integer indices")
	DeclFunc("Cell", Cell, "Single cell with given integer index (i, j, k)")
}

// geometrical shape for setting sample geometry
type Shape func(x, y, z float64) bool

// Ellipsoid with given diameters
func Ellipsoid(diamx, diamy, diamz float64) Shape {
	return func(x, y, z float64) bool {
		return sqr64(x/diamx)+sqr64(y/diamy)+sqr64(z/diamz) <= 0.25
	}
}

func Ellipse(diamx, diamy float64) Shape {
	return Ellipsoid(diamx, diamy, math.Inf(1))
}

func Circle(diam float64) Shape {
	return Cylinder(diam, math.Inf(1))
}

// cylinder along z.
func Cylinder(diam, height float64) Shape {
	return func(x, y, z float64) bool {
		return z <= height/2 && z >= -height/2 &&
			sqr64(x/diam)+sqr64(y/diam) <= 0.25
	}
}

// 3D Rectangular slab with given sides.
func Cuboid(sidex, sidey, sidez float64) Shape {
	return func(x, y, z float64) bool {
		rx, ry, rz := sidex/2, sidey/2, sidez/2
		return x < rx && x > -rx && y < ry && y > -ry && z < rz && z > -rz
	}
}

// 2D Rectangle with given sides.
func Rect(sidex, sidey float64) Shape {
	return func(x, y, z float64) bool {
		rx, ry := sidex/2, sidey/2
		return x < rx && x > -rx && y < ry && y > -ry
	}
}

// All cells with x-coordinate between a and b
func XRange(a, b float64) Shape {
	return func(x, y, z float64) bool {
		return x >= a && x < b
	}
}

// All cells with y-coordinate between a and b
func YRange(a, b float64) Shape {
	return func(x, y, z float64) bool {
		return y >= a && y < b
	}
}

// All cells with z-coordinate between a and b
func ZRange(a, b float64) Shape {
	return func(x, y, z float64) bool {
		return z >= a && z < b
	}
}

// Cell layers #a (inclusive) up to #b (exclusive).
func Layers(a, b int) Shape {
	Nz := Mesh().Size()[0]
	if a < 0 || a > Nz || b < 0 || b > Nz {
		log.Fatal("layers ", a, ":", b, " out of bounds (0 - ", Nz, ")")
	}
	c := Mesh().CellSize()[0]
	n := float64(Nz)
	z1 := ((float64(a) - n/2 - 0.5) * c)
	z2 := ((float64(b-1) - n/2 + 0.5) * c)
	return ZRange(z1, z2)
}

// Single cell with given index
func Cell(k, j, i int) Shape {
	c := Mesh().CellSize()
	n := Mesh().Size()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]
	x1 := float64(k)*c[2] - dx - c[2]/2
	y1 := float64(j)*c[1] - dy - c[1]/2
	z1 := float64(i)*c[0] - dz - c[0]/2
	x2 := float64(k)*c[2] - dx + c[2]/2
	y2 := float64(j)*c[1] - dy + c[1]/2
	z2 := float64(i)*c[0] - dz + c[0]/2
	return func(x, y, z float64) bool {
		return x > x1 && x < x2 &&
			y > y1 && y < y2 &&
			z > z1 && z < z2
	}
}

// The entire space.
func universe(x, y, z float64) bool {
	return true
}

// Transl returns a translated copy of the shape.
func (s Shape) Transl(dx, dy, dz float64) Shape {
	return func(x, y, z float64) bool {
		return s(x-dx, y-dy, z-dz)
	}
}

// Scale returns a scaled copy of the shape.
func (s Shape) Scale(sx, sy, sz float64) Shape {
	return func(x, y, z float64) bool {
		return s(x/sx, y/sy, z/sz)
	}
}

// Rotates the shape around the Z-axis, over θ radians.
func (s Shape) RotZ(θ float64) Shape {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) bool {
		x_ := x*cos + y*sin
		y_ := -x*sin + y*cos
		return s(x_, y_, z)
	}
}

// Rotates the shape around the Y-axis, over θ radians.
func (s Shape) RotY(θ float64) Shape {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) bool {
		x_ := x*cos + z*sin
		z_ := -x*sin + z*cos
		return s(x_, y, z_)
	}
}

// Rotates the shape around the X-axis, over θ radians.
func (s Shape) RotX(θ float64) Shape {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) bool {
		y_ := z*cos + y*sin
		z_ := -z*sin + y*cos
		return s(x, y_, z_)
	}
}

// Union of shapes a and b (logical OR).
func (a Shape) Add(b Shape) Shape {
	return func(x, y, z float64) bool {
		return a(x, y, z) || b(x, y, z)
	}
}

// Intersection of shapes a and b (logical AND).
func (a Shape) Intersect(b Shape) Shape {
	return func(x, y, z float64) bool {
		return a(x, y, z) && b(x, y, z)
	}
}

// Inverse (outside) of shape (logical NOT).
func (s Shape) Inverse() Shape {
	return func(x, y, z float64) bool {
		return !s(x, y, z)
	}
}

// Removes b from a (logical a AND NOT b)
func (a Shape) Sub(b Shape) Shape {
	return func(x, y, z float64) bool {
		return a(x, y, z) && !b(x, y, z)
	}
}

// Logical XOR of shapes a and b
func (a Shape) Xor(b Shape) Shape {
	return func(x, y, z float64) bool {
		A, B := a(x, y, z), b(x, y, z)
		return (A || B) && !(A && B)
	}
}

func sqr64(x float64) float64 { return x * x }
