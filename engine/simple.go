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

	//Torque.Memset(0, 0, 0)
	B_demag.AddTo(Torque)

	return nil
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

	m := M.Write() // will block until save done, pass rwmutex to heun??
	Solver.Step(m)
	cuda.Normalize(m)
	M.WriteDone()

	//util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", e.NSteps, e.undone, *e.Time, e.dt_si, err) // TODO: move
}

//func SetM(mx, my, mz float64) {
//	checkInited()
//	cuda.Memset(m, float32(mz), float32(my), float32(mx))
//	cuda.Normalize(m)
//}

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
