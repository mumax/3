package engine

import (
	"fmt"
	"slices"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var globalmesh_ data.Mesh // mesh for m and everything that has the same size

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetMesh", SetMesh, `Sets GridSize, CellSize and PBC at the same time`)
	DeclFunc("SetPBC", SetPBC, "Sets the number of repetitions in X,Y,Z to create periodic boundary "+
		"conditions. The number of repetitions determines the cutoff range for the demagnetization.")
}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh_
}

func arg(msg string, test bool) {
	if !test {
		panic(UserErr(msg + ": illegal arugment"))
	}
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
// TODO: dedup arguments from globals
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz int) {
	SetBusy(true)
	defer SetBusy(false)

	arg("GridSize", Nx > 0 && Ny > 0 && Nz > 0)
	arg("CellSize", cellSizeX > 0 && cellSizeY > 0 && cellSizeZ > 0)
	arg("PBC", pbcx >= 0 && pbcy >= 0 && pbcz >= 0)

	warnStr := "// WARNING: %s-axis is not 7-smooth. It has %d cells, with prime\n" +
		"//          factors %v, at least one of which is greater than 7.\n" +
		"//          Prime factors >7 may reduce performance significantly, and\n" +
		"//          prime factors >127 may cause a CUDA_ERROR_INVALID_VALUE error."
	if factorsx := primeFactors(Nx); slices.Max(factorsx) > 7 {
		util.Log(fmt.Sprintf(warnStr, "x", Nx, factorsx))
	}
	if factorsy := primeFactors(Ny); slices.Max(factorsy) > 7 {
		util.Log(fmt.Sprintf(warnStr, "y", Ny, factorsy))
	}
	if factorsz := primeFactors(Nz); slices.Max(factorsz) > 7 {
		util.Log(fmt.Sprintf(warnStr, "z", Nz, factorsz))
	}

	sizeChanged := globalmesh_.Size() != [3]int{Nx, Ny, Nz}
	cellSizeChanged := globalmesh_.CellSize() != [3]float64{cellSizeX, cellSizeY, cellSizeZ}
	pbc := []int{pbcx, pbcy, pbcz}

	if globalmesh_.Size() == [3]int{0, 0, 0} {
		// first time mesh is set
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.alloc()
		regions.alloc()
	} else {
		// here be dragons
		LogOut("resizing...")

		// free everything to trigger kernel recalculation, etc
		conv_.Free()
		conv_ = nil
		mfmconv_.Free()
		mfmconv_ = nil
		cuda.FreeBuffers()

		// resize everything
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		if sizeChanged || cellSizeChanged {
			M.resize()
			regions.resize()
			geometry.buffer.Free()
			geometry.buffer = data.NilSlice(1, Mesh().Size())
			geometry.setGeom(geometry.shape)

			// remove excitation extra terms if they don't fit anymore
			// up to the user to add them again
			B_ext.RemoveExtraTerms()
			J.RemoveExtraTerms()

			B_therm.noise.Free()
			B_therm.noise = nil
		}
	}
	lazy_gridsize = []int{Nx, Ny, Nz}
	lazy_cellsize = []float64{cellSizeX, cellSizeY, cellSizeZ}
	lazy_pbc = []int{pbcx, pbcy, pbcz}
}

func printf(f float64) float32 {
	return float32(f)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	lazy_gridsize []int
	lazy_cellsize []float64
	lazy_pbc      = []int{0, 0, 0}
)

func SetGridSize(Nx, Ny, Nz int) {
	lazy_gridsize = []int{Nx, Ny, Nz}
	if lazy_cellsize != nil {
		SetMesh(Nx, Ny, Nz, lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z], lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetCellSize(cx, cy, cz float64) {
	lazy_cellsize = []float64{cx, cy, cz}
	if lazy_gridsize != nil {
		SetMesh(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z], cx, cy, cz, lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetPBC(nx, ny, nz int) {
	lazy_pbc = []int{nx, ny, nz}
	if lazy_gridsize != nil && lazy_cellsize != nil {
		SetMesh(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z],
			lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z],
			lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

// check if mesh is set
func checkMesh() {
	if globalmesh_.Size() == [3]int{0, 0, 0} {
		panic("need to set mesh first")
	}
}
