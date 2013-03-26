package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
)

// User inputs
var (
	Aex   ScalFn
	Msat  ScalFn
	Alpha ScalFn
	Bext  VecFn = ConstVector(0, 0, 0)
	DMI   VecFn = ConstVector(0, 0, 0)
)

var (
	mesh   *data.Mesh
	solver *cuda.Heun
	Time   float64
	Solver *cuda.Heun
)

var (
	M, Torque       *Buffered
	B_demag, B_exch *Adder
)

func initialize() {

	M = NewBuffered(3, "m")

	demag_ := cuda.NewDemag(mesh)
	vol := data.NilSlice(3, mesh)
	B_demag = NewAdder("B_demag", func(dst *data.Slice) {
		m := M.Read()
		demag_.Exec(dst, m, vol, Mu0*Msat())
		M.ReadDone()
	})

	B_exch = NewAdder("B_exch", func(dst *data.Slice) {
		m := M.Read()
		cuda.AddExchange(dst, m, Aex(), Mu0*Msat())
		M.ReadDone()
	})

	Torque = NewBuffered(3, "torque")

	Solver = cuda.NewHeun(mesh, TorqueFn, 1e-15, Gamma0, &Time)
}

func TorqueFn() *data.Slice {

}

func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if mesh != nil {
		log.Fatal("mesh already set")
	}
	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", mesh)
	initialize()
}
