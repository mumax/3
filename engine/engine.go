package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/util"
	"log"
)

var (
	Aex   ScalFn
	Msat  ScalFn
	Alpha ScalFn
	Bext  VecFn = ConstVector(0, 0, 0)
	DMI   VecFn = ConstVector(0, 0, 0)
	Time  float64
)

var (
	mesh          *data.Mesh
	Solver        *cuda.Heun
	m, mx, my, mz *data.Slice
	buffer        *data.Slice // holds H_effective or torque
	vol           *data.Slice
	Demag         Quant
	Exch          Quant
	Table         autosave
)

// Evaluates all quantities, possibly saving them in the meanwhile.
func Eval() *data.Slice {
	// save M here
	cuda.Memset(buffer, 0, 0, 0) // Need this in case demag is output, then we really add to.
	Demag.AddTo(buffer)          // Does not add but sets, so it should be first.
	Exch.AddTo(buffer)
	bext := Bext()
	cuda.AddConst(buffer, float32(bext[Z]), float32(bext[Y]), float32(bext[X]))
	cuda.LLGTorque(buffer, m, buffer, float32(Alpha()))
	// save torque here
	return buffer
}

func initialize() {
	m = cuda.NewSlice(3, mesh)
	mx, my, mz = m.Comp(0), m.Comp(1), m.Comp(2)
	buffer = cuda.NewSlice(3, mesh)
	vol = data.NilSlice(1, mesh)
	Solver = cuda.NewHeun(m, Eval, 1e-15, Gamma0, &Time)

	demag := cuda.NewDemag(mesh)
	Demag = NewQuant("B_demag", func(dst *data.Slice) {
		demag.Exec(dst, m, vol, Mu0*Msat())
	})

	Exch = NewQuant("B_exch", func(dst *data.Slice) {
		cuda.AddExchange(dst, m, Aex(), Mu0*Msat()) // !! ADD TO DST, NOT H !
	})
}

func checkInited() {
	if mesh == nil {
		log.Fatal("need to set mesh first")
	}
}

func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if mesh != nil {
		log.Fatal("mesh already set")
	}
	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", mesh)
	initialize()
}

func SetM(mx, my, mz float64) {
	checkInited()
	cuda.Memset(m, float32(mz), float32(my), float32(mx))
	cuda.Normalize(m)
}

func Run(seconds float64) {
	checkInited()
	stop := Time + seconds
	for Time < stop {
		step()
	}
	util.DashExit()
}

func Steps(n int) {
	checkInited()
	for i := 0; i < n; i++ {
		step()
	}
	util.DashExit()
}

func step() {
	savetable()
	Solver.Step(m)
	cuda.Normalize(m)
}

const (
	Mu0    = mag.Mu0
	Gamma0 = mag.Gamma0
	X      = 0
	Y      = 1
	Z      = 2
)
