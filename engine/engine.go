// Package engine implements a high-level API. Most of the API is available through the script syntax as well.
package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
	"os"
	"runtime"
)

const VERSION = "mx3.0.11 α "

var UNAME = VERSION + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + "(" + runtime.Compiler + ")"

func init() {
	Table = *newTable("datatable") // output handle for tabular data (average magnetization etc.)
	World.Func("setgridsize", setGridSize, `Sets the number of cells for X,Y,Z`)
	World.Func("setcellsize", setCellSize, `Sets the X,Y,Z cell size in meters`)
	World.LValue("m", &M, `Reduced magnetization (unit length)`)
	World.ROnly("torque", &Torque)
	World.ROnly("B_eff", &B_eff)
	World.ROnly("table", &Table)

	hostname, _ := os.Hostname()
	GUI.SetValue("hostname", hostname)
}

var (
	M      magnetization // reduced magnetization (unit length)
	M_full SetterQuant   // non-reduced magnetization in T
	B_eff  SetterQuant   // effective field (T) output handle
	Torque SetterQuant   // total torque/γ0, in T
)

var Table DataTable

var (
	globalmesh data.Mesh
	itime      int                       // unique integer time stamp // TODO: revise
	Quants     = make(map[string]Getter) // INTERNAL maps quantity names to downloadable data. E.g. for rendering
)

func initialize() {
	M.init()
	FFTM.init()
	Quants["m"] = &M
	Quants["mFFT"] = &fftmPower{} // for the web interface we display FFT amplitude

	M_full = setter(3, Mesh(), "m_full", "T", func(dst *data.Slice, g bool) {
		msat, r := Msat.GetGPU()
		util.Assert(r == true)
		defer cuda.RecycleBuffer(msat)
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), M.buffer.Comp(c), msat)
		}
	})

	regions.init()
	Quants["regions"] = &regions

	Table.Add(&M)

	initDemag()
	initExchange()
	initAnisotropy()
	initBExt()

	// effective field
	B_eff = setter(3, Mesh(), "B_eff", "T", func(dst *data.Slice, cansave bool) {
		B_demag.set(dst, cansave)
		B_exch.addTo(dst, cansave)
		B_anis.addTo(dst, cansave)
		B_ext.addTo(dst) // TODO: cansave
	})
	Quants["B_eff"] = &B_eff

	// torque terms
	initLLTorque()
	initSTTorque()
	Torque = setter(3, Mesh(), "torque", "T", func(b *data.Slice, cansave bool) {
		LLTorque.set(b, cansave)
		STTorque.addTo(b, cansave)
	})
	Quants["torque"] = &Torque

	torquebuffer := cuda.NewSlice(3, Mesh())
	torqueFn := func(cansave bool) *data.Slice {
		itime++

		if cansave {
			//Table.arm(cansave)      // if table output needed, quantities marked for update
			notifySave(&M, cansave) // saves m if needed
			notifySave(&FFTM, cansave)
			// TODO: use list
		}

		Torque.set(torquebuffer, cansave)

		Table.touch(cansave) // all needed quantities are now up-to-date, save them
		return torquebuffer
	}
	Solver = *cuda.NewHeun(M.buffer, torqueFn, cuda.Normalize, 1e-15, Gamma0, &Time)
}

//func sanitycheck() {
//	if Msat() == 0 {
//		log.Fatal("Msat should be nonzero")
//	}
//}

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
	GUI.SetValue("nx", Nx)
	GUI.SetValue("ny", Ny)
	GUI.SetValue("nz", Nz)
	GUI.SetValue("cx", cellSizeX*1e9) // in nm
	GUI.SetValue("cy", cellSizeY*1e9)
	GUI.SetValue("cz", cellSizeZ*1e9)
	GUI.SetValue("wx", float64(Nx)*cellSizeX*1e9)
	GUI.SetValue("wy", float64(Ny)*cellSizeY*1e9)
	GUI.SetValue("wz", float64(Nz)*cellSizeZ*1e9)
	initialize()
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
		panic("need to set mesh first") //todo: fatal
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
}
