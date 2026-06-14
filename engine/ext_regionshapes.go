package engine

import (
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("ext_AllRegionShapes", ext_AllRegionShapes, "Returns a function that gives the shape of each region. Does not automatically update if the region is redefined or moved with Shift. Call as: allregions := ext_AllRegionShapes(); shape := allregions(i); on seperate lines. Note: double parentheses as ext_AllRegionShapes()(i) are not supported in mx3 scripts and must be split across two lines.")
}

func ext_AllRegionShapes() func(int) Shape {
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
