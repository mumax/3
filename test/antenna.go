package main

import (
	"github.com/mumax/3/data"
	. "github.com/mumax/3/engine"
	. "github.com/mumax/3/mainpkg"
	"math"
)

const Mu0 = 4 * math.Pi * 1e-7

func main() {

	Init()
	defer Close()

	Nx := 512
	Ny := 128
	Nz := 1

	cellsize := 5.0e-9
	SetGridSize(Nx, Ny, Nz)
	thickness := 40e-9
	//width := float64(Ny) * cellsize
	length := float64(Nx) * cellsize
	SetCellSize(cellsize, cellsize, thickness/float64(Nz))

	mask := data.NewSlice(3, Mesh().Size())
	wireX := -length * 0.45
	//wireY := 0.0
	wireZ := thickness * 5.0

	for h := 0; h < 10; h++ {
		for i := 0; i < Nx; i++ {
			for j := 0; j < Ny; j++ {
				r := Index2Coord(i, j, 0)
				r = r.Sub(Vector(wireX+float64(h)*cellsize, r.Y(), wireZ))

				B := Vector(0, 0, 0)
				current := Vector(0, 1, 0)
				B = r.Cross(current).Mul(Mu0 / (2 * math.Pi * math.Pow(r.Len(), 2)))

				mask.Set(0, i, j, 0, B.X())
				mask.Set(1, i, j, 0, B.Y())
				mask.Set(2, i, j, 0, B.Z())
			}
		}
	}

	data.MustWriteFile("mask.dump", mask, data.Meta{})
}
