package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"runtime"
	"time"
)

const VERSION = "mumax3.4.0"

var UNAME = VERSION + " " + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + " (" + runtime.Compiler + ")"

var StartTime = time.Now()

var (
	globalmesh data.Mesh     // mesh for m and everything that has the same size
	M          magnetization // reduced magnetization (unit length)
	B_eff      setter        // total effective field
)

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetPBC", SetPBC, `Sets number of repetitions in X,Y,Z`)
	DeclLValue("m", &M, `Reduced magnetization (unit length)`)
	B_eff.init(VECTOR, &globalmesh, "B_eff", "T", "Effective field", SetEffectiveField)
}

// Sets dst to the current effective field (T).
func SetEffectiveField(dst *data.Slice) {
	B_demag.Set(dst)  // set to B_demag...
	B_exch.AddTo(dst) // ...then add other terms
	B_anis.AddTo(dst)
	B_ext.AddTo(dst)
	B_therm.AddTo(dst)
}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if Nx <= 1 {
		util.Fatal("mesh size X should be > 1, have: ", Nx)
	}
	globalmesh = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbcxyz...)
	alloc()
}

// allocate m and regions buffer (after mesh is set)
func alloc() {
	M.alloc()
	regions.alloc()
}

func normalize(m *data.Slice) {
	cuda.Normalize(m, nil)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	gridsize []int
	cellsize []float64
	pbcxyz   []int
)

func SetGridSize(Nx, Ny, Nz int) {
	gridsize = []int{Nx, Ny, Nz}
	if cellsize != nil {
		SetMesh(Nx, Ny, Nz, cellsize[0], cellsize[1], cellsize[2])
	}
}

func SetCellSize(cx, cy, cz float64) {
	cellsize = []float64{cx, cy, cz}
	if gridsize != nil {
		SetMesh(gridsize[0], gridsize[1], gridsize[2], cx, cy, cz)
	}
}

func SetPBC(nx, ny, nz int) {
	if pbcxyz != nil {
		util.Fatal("PBC alread set")
	}
	if globalmesh.Size() != [3]int{0, 0, 0} {
		util.Fatal("PBC must be set before MeshSize and GridSize")
	}
	pbcxyz = []int{nx, ny, nz}
}

// check if mesh is set
func checkMesh() {
	if globalmesh.Size() == [3]int{0, 0, 0} {
		util.Fatal("need to set mesh first")
	}
}

// check if m is set
func checkM() {
	checkMesh()
	if M.Buffer().DevPtr(0) == nil {
		util.Fatal("need to initialize magnetization first")
	}
	if cuda.MaxVecNorm(M.Buffer()) == 0 {
		util.Fatal("need to initialize magnetization first")
	}
}

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {
	drainOutput()
	Table.flush()
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	//log.Println("Total memory allocation", memstats.TotalAlloc/(1024), "KiB")

	// debug. TODO: rm
	//	for n, p := range params {
	//		if u, ok := p.(interface {
	//			nUpload() int
	//		}); ok {
	//			log.Println(n, "\t:\t", u.nUpload(), "uploads")
	//		}
	//	}
}

// TODO
//func sanitycheck() {
//	if Msat() == 0 {
//		log.Fatal("Msat should be nonzero")
//	}
//}
