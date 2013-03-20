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
	mesh   *data.Mesh
	Solver *cuda.Heun
	m, h   *data.Slice
	vol    *data.Slice
	Demag  Quant
	Exch   Quant
)

func torque() *data.Slice {
	cuda.Memset(h, 0, 0, 0) // Need this in case demag is output, then we really add to.
	Demag.AddTo(h)          // Does not add but sets, so it should be first.
	Exch.AddTo(h)
	bext := Bext()
	cuda.AddConst(h, float32(bext[Z]), float32(bext[Y]), float32(bext[X]))

	cuda.LLGTorque(h, m, h, float32(Alpha()))
	return h
}

func initialize() {
	m = cuda.NewSlice(3, mesh)
	h = cuda.NewSlice(3, mesh)
	vol = data.NilSlice(1, mesh)

	Solver = cuda.NewHeun(m, torque, 1e-15, Gamma0, &Time)

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
