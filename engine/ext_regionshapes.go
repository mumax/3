package engine

import (
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("Region2Shape", Region2Shape, "Shape containing all cells belonging to the given region")
	DeclFunc("AllRegionShapes", AllRegionShapes, "Returns a function that gives the shape of each region")
}

func Region2Shape(region int) Shape {
	defRegionId(region)

	return func(x, y, z float64) bool {
		R := data.Vector{x, y, z}
		return regions.get(R) == region
	}
}

func AllRegionShapes() func(int) Shape {
	return func(region int) Shape {
		defRegionId(region)

		return func(x, y, z float64) bool {
			R := data.Vector{x, y, z}
			return regions.get(R) == region
		}
	}
}
