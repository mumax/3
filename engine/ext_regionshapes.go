package engine

import "github.com/mumax/3/data"

func init() {
	DeclFunc("Region2Shape", Region2Shape, "Shape containing all cells belonging to the given region")
	DeclFunc("AllRegionShapes", AllRegionShapes, "Returns a function that gives the shape of each region")
}

func Region2Shape(region int) Shape {
	defRegionId(region)
	mesh := Mesh()
	arr := regions.HostList()
	n := mesh.Size()
	d := mesh.CellSize()
	Lx, Ly, Lz := float64(n[X])*d[X], float64(n[Y])*d[Y], float64(n[Z])*d[Z]

	return func(x, y, z float64) bool {

		ix := int((x + 0.5*Lx) / d[X])
		iy := int((y + 0.5*Ly) / d[Y])
		iz := int((z + 0.5*Lz) / d[Z])

		if ix < 0 || ix >= n[X] || iy < 0 || iy >= n[Y] || iz < 0 || iz >= n[Z] {
			return false
		}

		i := data.Index(n, ix, iy, iz)
		return arr[i] == byte(region)
	}
}

func AllRegionShapes() func(int) Shape {
	mesh := Mesh()
	arr := regions.HostList()
	n := mesh.Size()
	d := mesh.CellSize()
	Lx, Ly, Lz := float64(n[X])*d[X], float64(n[Y])*d[Y], float64(n[Z])*d[Z]

	return func(region int) Shape {
		defRegionId(region)

		return func(x, y, z float64) bool {
			ix := int((x + 0.5*Lx) / d[X])
			iy := int((y + 0.5*Ly) / d[Y])
			iz := int((z + 0.5*Lz) / d[Z])

			if ix < 0 || ix >= n[X] || iy < 0 || iy >= n[Y] || iz < 0 || iz >= n[Z] {
				return false
			}

			i := data.Index(n, ix, iy, iz)
			return arr[i] == byte(region)
		}
	}
}

func ShiftShapeX(s Shape, dx int) Shape {
	if dx == 0 {
		return s
	}

	mesh := Mesh()
	d := mesh.CellSize()

	shift := float64(dx) * d[X]

	return func(x, y, z float64) bool {
		return s(x-shift, y, z)
	}
}

func ShiftShapeY(s Shape, dy int) Shape {
	if dy == 0 {
		return s
	}

	mesh := Mesh()
	d := mesh.CellSize()
	shift := float64(dy) * d[Y]

	return func(x, y, z float64) bool {
		return s(x, y-shift, z)
	}
}
