package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var globalmesh_ data.Mesh // mesh for m and everything that has the same size

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetPBC", SetPBC, `Sets number of repetitions in X,Y,Z`)
	DeclFunc("SetMesh", SetMesh, `Sets GridSize, CellSize and PBC in once`)
}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh_
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
// TODO: dedup arguments from globals
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz int) {
	SetBusy(true)
	defer SetBusy(false)

	prevSize := globalmesh_.Size()
	pbc := []int{pbcx, pbcy, pbcz}

	if Nx <= 1 {
		util.Fatal("mesh size X should be > 1, have: ", Nx)
	}

	if globalmesh_.Size() == [3]int{0, 0, 0} {
		// first time mesh is set
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.alloc()
		regions.alloc()
	} else {
		// here be dragons
		LogOutput("resizing")

		// free everything
		conv_.Free()
		conv_ = nil
		mfmconv_.Free()
		mfmconv_ = nil
		cuda.FreeBuffers()

		// resize everything
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.resize()
		regions.resize()
		geometry.buffer.Free()
		geometry.buffer = data.NilSlice(1, Mesh().Size())
		geometry.setGeom(geometry.shape)

		// remove excitation extra terms if they don't fit anymore
		// up to the user to add them again
		if Mesh().Size() != prevSize {
			B_ext.RemoveExtraTerms()
			J.RemoveExtraTerms()
		}
	}

}

func printf(f float64) float32 {
	return float32(f)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	gridsize []int
	cellsize []float64
	pbc      = []int{0, 0, 0}
)

func SetGridSize(Nx, Ny, Nz int) {
	gridsize = []int{Nx, Ny, Nz}
	if cellsize != nil {
		SetMesh(Nx, Ny, Nz, cellsize[X], cellsize[Y], cellsize[Z], pbc[X], pbc[Y], pbc[Z])
	}
}

func SetCellSize(cx, cy, cz float64) {
	cellsize = []float64{cx, cy, cz}
	if gridsize != nil {
		SetMesh(gridsize[X], gridsize[Y], gridsize[Z], cx, cy, cz, pbc[X], pbc[Y], pbc[Z])
	}
}

func SetPBC(nx, ny, nz int) {
	pbc = []int{nx, ny, nz}
	if gridsize != nil && cellsize != nil {
		SetMesh(gridsize[X], gridsize[Y], gridsize[Z], cellsize[X], cellsize[Y], cellsize[Z], pbc[X], pbc[Y], pbc[Z])
	}
}

// check if mesh is set
func checkMesh() {
	if globalmesh_.Size() == [3]int{0, 0, 0} {
		util.Fatal("need to set mesh first")
	}
}
