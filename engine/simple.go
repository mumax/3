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
	mesh    *data.Mesh
	Solver  *cuda.Heun
	inited  bool
	m, h    *data.Slice
	vol     *data.Slice
	demag   *cuda.DemagConvolution
	addExch adder
)

func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if mesh != nil {
		log.Fatal("mesh already set")
	}
	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", mesh)

	m = cuda.NewSlice(3, mesh)
	h = cuda.NewSlice(3, mesh)
	vol = data.NilSlice(1, mesh)

	Solver = cuda.NewHeun(m, torque, 1e-15, Gamma0, &Time)

	demag = cuda.NewDemag(mesh)

	addExch = addOutput(func(h *data.Slice) {
		cuda.AddExchange(h, m, Aex(), Mu0*Msat())
	})

}

func initialize() {
	if mesh == nil {
		log.Fatal("need to set mesh first")
	}
}

func SetM(mx, my, mz float64) {
	initialize()
	cuda.Memset(m, float32(mz), float32(my), float32(mx))
}

func Run(seconds float64) {
	initialize()
	stop := Time + seconds
	for Time < stop {
		Solver.Step(m)
	}
}

func torque(m *data.Slice) *data.Slice {
	msat := Msat()
	demag.Exec(h, m, vol, Mu0*msat)

	addExch(h)

	bext := Bext()
	cuda.AddConst(h, float32(bext[Z]), float32(bext[Y]), float32(bext[X]))

	cuda.LLGTorque(h, m, h, float32(Alpha()))
	return h
}

type adder func(dst *data.Slice)

func addOutput(f adder) adder {
	return func(dst *data.Slice) {
		// TODO:
		// if need output:
		// add to zeroed buffer, output buffer (async), add buffer to dst
		// pipe buffers to/from output goroutine
		// else:
		f(dst)
	}
}

const (
	Mu0    = mag.Mu0
	Gamma0 = mag.Gamma0
	X      = 0
	Y      = 1
	Z      = 2
)
