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
	B_demag, B_exch *adder
)

func initialize() {

	M = NewBuffered(3, "m")
	Torque = NewBuffered(3, "torque")

	demag_ := cuda.NewDemag(mesh)
	vol := data.NilSlice(1, mesh)
	B_demag = newAdder("B_demag", func(dst *data.Slice) {
		m := M.Read()
		demag_.Exec(dst, m, vol, Mu0*Msat())
		M.ReadDone()
	})

	B_exch = newAdder("B_exch", func(dst *data.Slice) {
		m := M.Read()
		cuda.AddExchange(dst, m, Aex(), Mu0*Msat())
		M.ReadDone()
	})

	Solver = cuda.NewHeun(&M.Synced, TorqueFn, 1e-15, Gamma0, &Time)
}

func TorqueFn(good bool) *data.Synced {

	if good {
		M.Touch() // saves if needed
	}

	// Effective field
	Buf := Torque // first use torque buffer for effective field.
	Buf.Memset(0, 0, 0)
	B_demag.addTo(Buf) // properly locks and outputs if needed
	B_exch.addTo(Buf)

	// Torque
	b := Buf.Write() // B_eff, to be overwritten by torque.
	m := M.Read()
	cuda.LLGTorque(b, m, b, float32(Alpha()))
	M.ReadDone()
	Buf.WriteDone()

	if good {
		Torque.Touch() // saves if needed
	}

	return &Buf.Synced
}

func Run(seconds float64) {
	log.Println("run for", seconds, "s")
	checkInited()
	stop := Time + seconds
	defer util.DashExit()
	for Time < stop {
		step()
	}
}

func Steps(n int) {
	log.Println("run for", n, "steps")
	checkInited()
	defer util.DashExit()
	for i := 0; i < n; i++ {
		step()
	}
}

func step() {

	Solver.Step()
	M.Normalize()

	s := Solver
	util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, *s.Time, s.Dt_si, s.LastErr)
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
