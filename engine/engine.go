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

// Accessible quantities
var (
	M, B_eff, Torque, B_demag, B_exh Handle
	Time                             float64
	Solver                           *cuda.Heun
)

// 
var (
	mesh                      *data.Mesh
	solver                    *cuda.Heun
	m, b_eff, torque, b_demag *buffered
	b_exch, b_ext             *adder
	vol                       *data.Slice
)

func initialize() {

	// these 2 GPU arrays are re-used to stored various quantities.
	arr1, arr2 := cuda.NewSynced(3, mesh), cuda.NewSynced(3, mesh)

	m = newBuffered(arr1, "m", nil)
	M = m

	b_eff = newBuffered(arr2, "B_eff", nil)
	B_eff = b_eff

	vol = data.NilSlice(1, mesh)

	demag_ := cuda.NewDemag(mesh)
	b_demag = newBuffered(arr2, "B_demag", func(b *data.Slice) {
		m_ := m.Read()
		demag_.Exec(b, m_, vol, Mu0*Msat())
		m.ReadDone()
	})
	B_demag = b_demag

	b_exch = newAdder("B_exch", func(dst *data.Slice) {
		m_ := m.Read()
		cuda.AddExchange(dst, m_, Aex(), Mu0*Msat())
		m.ReadDone()
	})
	B_exh = b_exch

	b_ext = newAdder("B_ext", func(dst *data.Slice) {
		bext := B_ext()
		cuda.AddConst(dst, float32(bext[2]), float32(bext[1]), float32(bext[0]))
	})

	torque = newBuffered(arr2, "torque", func(b *data.Slice) {
		m_ := m.Read()
		cuda.LLGTorque(b, m_, b, float32(Alpha()))
		m.ReadDone()
	})
	Torque = torque

	torqueFn := func(good bool) *data.Synced {
		m.touch(good) // saves if needed
		b_demag.update(good)
		b_exch.addTo(b_eff, good)
		b_ext.addTo(b_eff, good)
		torque.update(good)
		return torque.Synced
	}

	Solver = cuda.NewHeun(m.Synced, torqueFn, 1e-15, Gamma0, &Time)
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
