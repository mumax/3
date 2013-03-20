package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
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
	demag  *cuda.DemagConvolution
	exch   Quant
)

func torque() *data.Slice {
	msat := Msat()
	demag.Exec(h, m, vol, Mu0*msat)

	exch.AddTo(h)

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

	demag = cuda.NewDemag(mesh)

	exch = NewQuant("B_ex", func(dst *data.Slice) {
		cuda.AddExchange(h, m, Aex(), Mu0*Msat())
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
