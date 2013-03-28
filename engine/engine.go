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
	B_ext VecFn = ConstVector(0, 0, 0)
	DMI   VecFn = ConstVector(0, 0, 0)
)

var (
	mesh   *data.Mesh
	solver *cuda.Heun
	Time   float64
	Solver *cuda.Heun
)

var (
	m, b_eff, torque                 *buffered
	b_demag, b_exch                  *adder
	M, B_eff, Torque, B_demag, B_exh Handle
)

func initialize() {

	m = newBuffered(3, "m")
	M = m

	torque = newBuffered(3, "torque")
	Torque = torque

	b_eff = &buffered{Synced: torque.Synced} // shares storages with torque, but has separate autosave
	b_eff.name = "B_eff"
	B_eff = b_eff

	demag_ := cuda.NewDemag(mesh)
	vol := data.NilSlice(1, mesh)
	b_demag = newAdder("B_demag", func(dst *data.Slice) {
		m_ := m.Read()
		demag_.Exec(dst, m_, vol, Mu0*Msat())
		m.ReadDone()
	})
	B_demag = b_demag

	b_exch = newAdder("B_exch", func(dst *data.Slice) {
		m_ := m.Read()
		cuda.AddExchange(dst, m_, Aex(), Mu0*Msat())
		m.ReadDone()
	})
	B_exh = b_exch

	Solver = cuda.NewHeun(&m.Synced, torqueFn, 1e-15, Gamma0, &Time)
}

func torqueFn(good bool) *data.Synced {

	m.touch(good) // saves if needed

	// Effective field
	b_eff.memset(0, 0, 0)      // !! although demag overwrites, must be zero in case we save...
	b_demag.addTo(b_eff, good) // !! this one has to be the first addTo
	b_exch.addTo(b_eff, good)
	b_eff.touch(good)
	addB_ext(b_eff)

	// Torque
	b := torque.Write() // B_eff, to be overwritten by torque.
	m_ := m.Read()
	cuda.LLGTorque(b, m_, b, float32(Alpha()))
	m.ReadDone()
	torque.WriteDone()
	torque.touch(good) // saves if needed

	return &torque.Synced
}

func addB_ext(Dst *buffered) {
	bext := B_ext()
	dst := Dst.Write()
	cuda.AddConst(dst, float32(bext[2]), float32(bext[1]), float32(bext[0]))
	Dst.WriteDone()
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
	s := Solver
	s.Step()
	m.normalize()
	util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, *s.Time, s.Dt_si, s.LastErr)
}

func SetM(mx, my, mz float32) {
	checkInited()
	m.memset(mx, my, mz)
	m.normalize()
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
