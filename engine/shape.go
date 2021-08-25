package engine

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("Ellipsoid", Ellipsoid, "3D Ellipsoid with axes in meter")
	DeclFunc("Ellipse", Ellipse, "2D Ellipse with axes in meter")
	DeclFunc("Cone", Cone, "3D Cone with diameter and height in meter. The top of the cone points in the +z direction.")
	DeclFunc("Cylinder", Cylinder, "3D Cylinder with diameter and height in meter")
	DeclFunc("Circle", Circle, "2D Circle with diameter in meter")
	DeclFunc("Cuboid", Cuboid, "Cuboid with sides in meter")
	DeclFunc("Rect", Rect, "2D rectangle with size in meter")
	DeclFunc("Triangle", Triangle, "Equilateral triangle with side in meter")
	DeclFunc("Square", Square, "2D square with size in meter")
	DeclFunc("XRange", XRange, "Part of space between x1 (inclusive) and x2 (exclusive), in meter")
	DeclFunc("YRange", YRange, "Part of space between y1 (inclusive) and y2 (exclusive), in meter")
	DeclFunc("ZRange", ZRange, "Part of space between z1 (inclusive) and z2 (exclusive), in meter")
	DeclFunc("Layers", Layers, "Part of space between cell layer1 (inclusive) and layer2 (exclusive), in integer indices")
	DeclFunc("Layer", Layer, "Single layer (along z), by integer index starting from 0")
	DeclFunc("Universe", Universe, "Entire space")
	DeclFunc("Cell", Cell, "Single cell with given integer index (i, j, k)")
	DeclFunc("ImageShape", ImageShape, "Use black/white image as shape")
	DeclFunc("GrainRoughness", GrainRoughness, "Grainy surface with different heights per grain "+
		"with a typical grain size (first argument), minimal height (second argument), and maximal "+
		"height (third argument). The last argument is a seed for the random number generator.")
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

// 3D Cone with the vertex down.
func Cone(diam, height float64) Shape {
	return func(x, y, z float64) bool {
		return z >= 0 && sqr64(x/diam)+sqr64(y/diam)+0.25*sqr64(z/height) <= 0.25
	}
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

// Equilateral triangle with given sides.
func Triangle(side float64) Shape {
	return func(x, y, z float64) bool {
		c := math.Sqrt(3)
		return y > -side/(2*c) && y < x*c+side/c && y < -x*c+side/c
	}
}

// 2D square with given side.
func Square(side float64) Shape {
	return Rect(side, side)
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
	Nz := Mesh().Size()[Z]
	if a < 0 || a > Nz || b < 0 || b < a {
		util.Fatal("layers ", a, ":", b, " out of bounds (0 - ", Nz, ")")
	}
	c := Mesh().CellSize()[Z]
	z1 := Index2Coord(0, 0, a)[Z] - c/2
	z2 := Index2Coord(0, 0, b)[Z] - c/2
	return ZRange(z1, z2)
}

func Layer(index int) Shape {
	return Layers(index, index+1)
}

// Single cell with given index
func Cell(ix, iy, iz int) Shape {
	c := Mesh().CellSize()
	pos := Index2Coord(ix, iy, iz)
	x1 := pos[X] - c[X]/2
	y1 := pos[Y] - c[Y]/2
	z1 := pos[Z] - c[Z]/2
	x2 := pos[X] + c[X]/2
	y2 := pos[Y] + c[Y]/2
	z2 := pos[Z] + c[Z]/2
	return func(x, y, z float64) bool {
		return x > x1 && x < x2 &&
			y > y1 && y < y2 &&
			z > z1 && z < z2
	}
}

func Universe() Shape {
	return universe
}

// The entire space.
func universe(x, y, z float64) bool {
	return true
}

func ImageShape(fname string) Shape {
	r, err1 := httpfs.Open(fname)
	CheckRecoverable(err1)
	defer r.Close()
	img, _, err2 := image.Decode(r)
	CheckRecoverable(err2)

	width := img.Bounds().Max.X
	height := img.Bounds().Max.Y

	// decode image into bool matrix for fast pixel lookup
	inside := make([][]bool, height)
	for iy := range inside {
		inside[iy] = make([]bool, width)
	}
	for iy := 0; iy < height; iy++ {
		for ix := 0; ix < width; ix++ {
			r, g, b, a := img.At(ix, height-1-iy).RGBA()
			if a > 128 && r+g+b < (0xFFFF*3)/2 {
				inside[iy][ix] = true
			}
		}
	}

	// stretch the image onto the gridsize
	c := Mesh().CellSize()
	cx, cy := c[X], c[Y]
	N := Mesh().Size()
	nx, ny := float64(N[X]), float64(N[Y])
	w, h := float64(width), float64(height)
	return func(x, y, z float64) bool {
		ix := int((w/nx)*(x/cx) + 0.5*w)
		iy := int((h/ny)*(y/cy) + 0.5*h)
		if ix < 0 || ix >= width || iy < 0 || iy >= height {
			return false
		} else {
			return inside[iy][ix]
		}
	}
}

func GrainRoughness(grainsize, zmin, zmax float64, seed int) Shape {
	t := newTesselation(grainsize, 256, int64(seed))
	return func(x, y, z float64) bool {
		if z <= zmin {
			return true
		}
		if z >= zmax {
			return false
		}
		r := t.RegionOf(x, y, z)
		return (z-zmin)/(zmax-zmin) < (float64(r) / 256)
	}
}

// Transl returns a translated copy of the shape.
func (s Shape) Transl(dx, dy, dz float64) Shape {
	return func(x, y, z float64) bool {
		return s(x-dx, y-dy, z-dz)
	}
}

// Infinitely repeats the shape with given period in x, y, z.
// A period of 0 or infinity means no repetition.
func (s Shape) Repeat(periodX, periodY, periodZ float64) Shape {
	return func(x, y, z float64) bool {
		return s(fmod(x, periodX), fmod(y, periodY), fmod(z, periodZ))
	}
}

func fmod(a, b float64) float64 {
	if b == 0 || math.IsInf(b, 1) {
		return a
	}
	if math.Abs(a) > b/2 {
		return sign(a) * (math.Mod(math.Abs(a+b/2), b) - b/2)
	} else {
		return a
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
		x_ := x*cos - z*sin
		z_ := x*sin + z*cos
		return s(x_, y, z_)
	}
}

// Rotates the shape around the X-axis, over θ radians.
func (s Shape) RotX(θ float64) Shape {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) bool {
		y_ := y*cos + z*sin
		z_ := -y*sin + z*cos
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
