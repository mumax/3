package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"log"
)

var (
	Aex   Scal
	Msat  Scal
	Alpha Scal
	Bext  Vec = Vector(0, 0, 0)
	DMI   Vec = Vector(0, 0, 0)
)

var (
	mesh   *data.Mesh
	solver *cuda.Heun
	inited bool
	m, h   *data.Slice
	vol    *data.Slice
	demag  *cuda.DemagConvolution
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
	solver = cuda.NewHeun(m, torque, cuda.Normalize, 1e-15, Gamma0)
	demag = cuda.NewDemag(mesh)
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
	solver.Advance(seconds)
}

func torque(m *data.Slice, t float64) *data.Slice {
	msat := Msat.Val(t)
	demag.Exec(h, m, vol, Mu0*msat)
	cuda.AddExchange(h, m, Aex.Val(t), Mu0*msat)
	bext := Bext.Val(t)
	cuda.AddConst(h, float32(bext[Z]), float32(bext[Y]), float32(bext[X]))
	cuda.LLGTorque(h, m, h, float32(Alpha.Val(t)))
	return h
}

const (
	Mu0    = mag.Mu0
	Gamma0 = mag.Gamma0
	X      = 0
	Y      = 1
	Z      = 2
)
