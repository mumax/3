package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"log"
	"runtime"
)

const VERSION = "mumax3.0.11 α "

var UNAME = VERSION + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + "(" + runtime.Compiler + ")"

var (
	globalmesh    data.Mesh // mesh for m and everything that has the same size
	M             buffered  // reduced magnetization (unit length)
	B_eff, Torque setter
	Table         = *newTable("datatable") // output handle for tabular data (average magnetization etc.)
)

func init() {
	DeclFunc("setgridsize", setGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("setcellsize", setCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclROnly("table", &Table, `Provides methods for tabular output`)

	// magnetization
	M.init(3, "m", "", `Reduced magnetization (unit length)`, &globalmesh)

	// effective field
	B_eff.init(3, &globalmesh, "B_eff", "T", "Effective field", func(dst *data.Slice) {
		B_demag.set(dst)
		B_exch.addTo(dst)
		B_anis.addTo(dst)
		B_ext.addTo(dst)
	})

	// torque inited in torque.go
}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh
}

func WorldSize() [3]float64 {
	w := Mesh().WorldSize()
	return [3]float64{w[2], w[1], w[0]} // swaps XYZ
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if Nx <= 1 {
		log.Fatal("mesh size X should be > 1, have: ", Nx)
	}
	globalmesh = *data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", Mesh().UserString())
	alloc()
}

// allocate m and regions buffer (after mesh is set)
func alloc() {
	M.alloc()
	regions.alloc()
	Solver = *cuda.NewHeun(M.buffer, Torque.set, normalize, 1e-15, mag.Gamma0, &Time)
	Table.Add(MakeAvg(&M))
	vol = data.NilSlice(1, Mesh())
}

func normalize(m *data.Slice) {
	cuda.Normalize(m, nil)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	gridsize []int
	cellsize []float64
)

func setGridSize(Nx, Ny, Nz int) {
	gridsize = []int{Nx, Ny, Nz}
	if cellsize != nil {
		SetMesh(Nx, Ny, Nz, cellsize[0], cellsize[1], cellsize[2])
	}
}

func setCellSize(cx, cy, cz float64) {
	cellsize = []float64{cx, cy, cz}
	if gridsize != nil {
		SetMesh(gridsize[0], gridsize[1], gridsize[2], cx, cy, cz)
	}
}

// check if mesh is set
func checkMesh() {
	if globalmesh.Size() == [3]int{0, 0, 0} {
		log.Panic("need to set mesh first")
	}
}

// check if m is set
func checkM() {
	checkMesh()
	if M.buffer.DevPtr(0) == nil {
		log.Fatal("need to initialize magnetization first")
	}
	if cuda.MaxVecNorm(M.buffer) == 0 {
		log.Fatal("need to initialize magnetization first")
	}
}

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {
	log.Println("shutting down")
	drainOutput()
	Table.flush()

	// debug. TODO: rm
	for n, p := range params {
		if u, ok := p.(interface {
			NUpload() int
		}); ok {
			log.Println(n, "\t:\t", u.NUpload(), "uploads")
		}
	}
}

//func sanitycheck() {
//	if Msat() == 0 {
//		log.Fatal("Msat should be nonzero")
//	}
//}
