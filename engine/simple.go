package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
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
	Torque = NewBuffered(3, "torque")

	demag_ := cuda.NewDemag(mesh)
	vol := data.NilSlice(1, mesh)
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

	Solver = cuda.NewHeun(&M.Synced, TorqueFn, 1e-15, Gamma0, &Time)
}

func TorqueFn(good bool) *data.Synced {

	Torque.Memset(0, 0, 0)
	B_demag.AddTo(Torque)

	return &Torque.Synced
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
	M.Touch() // saves if needed

	Solver.Step()
	M.Normalize()

	//util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", e.NSteps, e.undone, *e.Time, e.dt_si, err) // TODO: move
}

func SetM(mx, my, mz float32) {
	checkInited()
	M.Memset(mx, my, mz)
	M.Normalize()
}

func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if mesh != nil {
		log.Fatal("mesh already set")
	}
	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", mesh)
	initialize()
}

func checkInited() {
	if mesh == nil {
		log.Fatal("need to set mesh first")
	}
}
