package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var globalmesh_ data.Mesh // mesh for m and everything that has the same size

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh_
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
// TODO: dedup arguments from globals
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64, pbc []int) {
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
		//mfmconv_.Free() // TODO
		//mfmconv = nil
		cuda.FreeBuffers()

		// resize everything
		newSize := [3]int{Nx, Ny, Nz}
		M.resize(newSize)
		regions.resize(newSize)

		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)

		geometry.buffer.Free()
		geometry.buffer = data.NilSlice(1, newSize)
		geometry.setGeom(geometry.shape) // uses global mesh
	}

	//Log("SetMesh", &globalmesh_)
	GUI.Set("nx", Nx)
	GUI.Set("ny", Ny)
	GUI.Set("nz", Nz)
	GUI.Set("cx", cellSizeX*1e9)
	GUI.Set("cy", cellSizeY*1e9)
	GUI.Set("cz", cellSizeZ*1e9)
	p := Mesh().PBC()
	GUI.Set("px", p[X])
	GUI.Set("py", p[Y])
	GUI.Set("pz", p[Z])
	w := Mesh().WorldSize()
	GUI.Set("wx", w[X]*1e9)
	GUI.Set("wy", w[Y]*1e9)
	GUI.Set("wz", w[Z]*1e9)
	GUI.Set("display", "/render/") // force display update
	GUI.Attr("renderLayer", "max", Nz)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	gridsize []int
	cellsize []float64
	pbc      []int
)

func SetGridSize(Nx, Ny, Nz int) {
	gridsize = []int{Nx, Ny, Nz}
	if cellsize != nil {
		SetMesh(Nx, Ny, Nz, cellsize[X], cellsize[Y], cellsize[Z], pbc)
	}
}

func SetCellSize(cx, cy, cz float64) {
	cellsize = []float64{cx, cy, cz}
	if gridsize != nil {
		SetMesh(gridsize[X], gridsize[Y], gridsize[Z], cx, cy, cz, pbc)
	}
}

func SetPBC(nx, ny, nz int) {
	pbc = []int{nx, ny, nz}
	if gridsize != nil && cellsize != nil {
		SetMesh(gridsize[X], gridsize[Y], gridsize[Z], cellsize[X], cellsize[Y], cellsize[Z], pbc)
	}
}

// check if mesh is set
func checkMesh() {
	if globalmesh_.Size() == [3]int{0, 0, 0} {
		util.Fatal("need to set mesh first")
	}
}
