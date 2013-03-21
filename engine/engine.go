package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/util"
	"sync"
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
	mesh          *data.Mesh
	Solver        *cuda.Heun
	m, mx, my, mz *data.Slice
	buffer        *data.Slice // holds H_effective or torque
	vol           *data.Slice
	demag         addFn
	exch          addFn
)

var (
	M, B_demag, B_exch, B_eff, Torque *Handle
)

type Handle struct{
	autosave
}

func (h *Handle) Need() bool {
	return h.needSave()
}

func (h *Handle) Send(s *data.Slice) {
		log.Println("need save:", h.Need())
		log.Println("\nsave", h.name)

		saveAndRecycle(s , h.fname(), Time)

		h.saved()
}

func newHandle(name string)*Handle{
	var h Handle
	h.name = name
	return &h	
}

// Evaluates all quantities, possibly saving them in the meanwhile.
func Eval() *data.Slice { // todo: output bool
	//doOutput := Solver.GoodStep

	output(m, M)

	cuda.Memset(buffer, 0, 0, 0)         // Need this in case demag is output, then we really add to.
	addAndOutput(buffer, demag, B_demag) // Does not add but sets, so it should be first.
	addAndOutput(buffer, exch, B_exch)

	bext := Bext()
	cuda.AddConst(buffer, float32(bext[Z]), float32(bext[Y]), float32(bext[X]))
	output(buffer, B_eff)

	cuda.LLGTorque(buffer, m, buffer, float32(Alpha()))
	output(buffer, Torque)

	return buffer
}

func output(s *data.Slice, h *Handle) {
	if h.Need(){h.Send(buffer)}
}

type addFn func(dst *data.Slice) // calculates quantity and add result to dst

var(
 gpubuf *data.Slice
 gpubuflock sync.Mutex
)

func outputBuffer()*data.Slice{
	if gpubuf == nil{
		gpubuf = cuda.NewSlice(3, m.Mesh())
	}
	return gpubuf
}

func addAndOutput(dst *data.Slice, addTo addFn, h *Handle) {
	if h.Need() {
		gpubuflock.Lock()
		buffer := gpubuf
		addTo(buffer)
		h.Send(buffer, LOCK?)
		cuda.Madd2(dst, dst, buffer, 1, 1)
		output(buffer, h)
	} else {
		addTo(dst)
	}
}

func initialize() {
	m = cuda.NewSlice(3, mesh)
	M = newHandle("m")

	mx, my, mz = m.Comp(0), m.Comp(1), m.Comp(2)
	buffer = cuda.NewSlice(3, mesh)
	vol = data.NilSlice(1, mesh)
	Solver = cuda.NewHeun(m, Eval, 1e-15, Gamma0, &Time)

	demag_ := cuda.NewDemag(mesh)
	demag = func(dst *data.Slice) {
		demag_.Exec(dst, m, vol, Mu0*Msat())
	}
	B_demag = newHandle("B_demag")

	exch = func(dst *data.Slice) {
		cuda.AddExchange(dst, m, Aex(), Mu0*Msat())
	}
	B_exch = newHandle("B_exch")

	B_eff = newHandle("B_eff")

	Torque = newHandle("torque")
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
	//savetable()
	Solver.Step(m)
	//util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", e.NSteps, e.undone, *e.Time, e.dt_si, err) // TODO: move
	cuda.Normalize(m)
}

const (
	Mu0    = mag.Mu0
	Gamma0 = mag.Gamma0
	X      = 0
	Y      = 1
	Z      = 2
)
