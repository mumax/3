package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	cuda5 "github.com/barnex/cuda5/cuda"
	"log"
	"time"
)

// User inputs
var (
	Aex          ScalFn         = Const(0) // Exchange stiffness in J/m
	ExchangeMask *StaggeredMask            // Mask that scales Aex/Msat between cells.
	KuMask       *Mask
	Msat         ScalFn = Const(0)             // Saturation magnetization in A/m
	Alpha        ScalFn = Const(0)             // Damping constant
	B_ext        VecFn  = ConstVector(0, 0, 0) // External field in T
	DMI          ScalFn = Const(0)             // Dzyaloshinskii-Moriya vector in J/m²
	Ku1          VecFn  = ConstVector(0, 0, 0) // Uniaxial anisotropy vector in J/m³
	Xi           ScalFn = Const(0)             // Non-adiabaticity of spin-transfer-torque
	SpinPol      ScalFn = Const(1)             // Spin polarization of electrical current
	J            VecFn  = ConstVector(0, 0, 0) // Electrical current density
)

// Accessible quantities
var (
	M                 buffered   // reduced magnetization (unit length)
	AvgM              *scalar    // average magnetization
	B_eff             *setter    // effective field (T) output handle
	B_demag           *setter    // demag field (T) output handle
	B_dmi             *adder     // demag field (T) output handle
	B_exch            *adder     // exchange field (T) output handle
	B_uni             *adder     // field due to uniaxial anisotropy output handle
	STTorque          *adder     // spin-transfer torque output handle
	LLGTorque, Torque *setter    // torque/gamma0, in Tesla
	Table             *DataTable // output handle for tabular data (average magnetization etc.)
	Time              float64    // time in seconds  // todo: hide? setting breaks autosaves
	Solver            *cuda.Heun
)

// hidden quantities
var (
	globalmesh   data.Mesh
	torquebuffer *data.Slice
	vol          *data.Slice
	postStep     []func() // called on after every time step
	extFields    []extField
	itime        int //unique integer time stamp
)

func global_mesh() *data.Mesh {
	checkInited()
	return &globalmesh
}

// Add an additional space-dependent field to B_ext.
// The field is mask * multiplier, where mask typically contains space-dependent scaling values of the order of 1.
// multiplier can be time dependent.
// TODO: extend API (set one component, construct masks or read from file). Also for current.
func AddExtField(mask *data.Slice, multiplier ScalFn) {
	m := cuda.GPUCopy(mask)
	extFields = append(extFields, extField{m, multiplier})
}

type extField struct {
	mask *data.Slice
	mul  ScalFn
}

func initialize() {

	// these 2 GPU arrays are re-used to stored various quantities.
	torquebuffer = cuda.NewSlice(3, global_mesh())

	// cell volumes currently unused
	vol = data.NilSlice(1, global_mesh())

	// magnetization
	M = *newBuffered(cuda.NewSlice(3, global_mesh()), "m", "")
	quants["m"] = &M
	AvgM = newScalar(3, "m", "", func() []float64 {
		return average(&M)
	})

	// data table
	Table = newTable("datatable")
	Table.Add(AvgM)

	// demag field
	demag_ := cuda.NewDemag(global_mesh())
	B_demag = newSetter(3, global_mesh(), "B_demag", "T", func(b *data.Slice, good bool) {
		demag_.Exec(b, M.buffer, vol, Mu0*Msat()) //TODO: consistent msat or bsat
	})
	quants["B_demag"] = B_demag

	// exchange field
	B_exch = newAdder(3, global_mesh(), "B_exch", "T", func(dst *data.Slice) {
		cuda.AddExchange(dst, M.buffer, ExchangeMask.buffer, Aex(), Msat())
	})
	quants["B_exch"] = B_exch

	ExchangeMask = newStaggeredMask(global_mesh(), "exchangemask", "")
	quants["exchangemask"] = ExchangeMask

	// Dzyaloshinskii-Moriya field
	B_dmi = newAdder(3, global_mesh(), "B_dmi", "T", func(dst *data.Slice) {
		d := DMI()
		if d != 0 {
			cuda.AddDMI(dst, M.buffer, d, Msat())
		}
	})
	quants["B_dmi"] = B_dmi

	// uniaxial anisotropy
	B_uni = newAdder(3, global_mesh(), "B_uni", "T", func(dst *data.Slice) {
		ku1 := Ku1() // in J/m3
		if ku1 != [3]float64{0, 0, 0} {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, KuMask.buffer, ku1[2], ku1[1], ku1[0], Msat())
		}
	})
	quants["B_uni"] = B_uni

	KuMask = newMask(3, global_mesh(), "kumask", "")
	quants["KuMask"] = KuMask

	// external field
	b_ext := newAdder(3, global_mesh(), "B_ext", "T", func(dst *data.Slice) {
		bext := B_ext()
		cuda.AddConst(dst, float32(bext[2]), float32(bext[1]), float32(bext[0]))
		for _, f := range extFields {
			cuda.Madd2(dst, dst, f.mask, 1, float32(f.mul()))
		}
	})
	//quants["B_ext"] = B_ext

	// effective field
	B_eff = newSetter(3, global_mesh(), "B_eff", "T", func(dst *data.Slice, good bool) {
		B_demag.set(dst, good)
		B_exch.addTo(dst, good)
		B_dmi.addTo(dst, good)
		B_uni.addTo(dst, good)
		b_ext.addTo(dst, good)
	})
	quants["B_eff"] = B_eff

	// llg torque
	LLGTorque = newSetter(3, global_mesh(), "llgtorque", "T", func(b *data.Slice, good bool) {
		B_eff.set(b, good)
		cuda.LLGTorque(b, M.buffer, b, float32(Alpha()))
	})
	quants["llgtorque"] = LLGTorque

	// spin-transfer torque
	STTorque = newAdder(3, global_mesh(), "sttorque", "T", func(dst *data.Slice) {
		j := J()
		if j != [3]float64{0, 0, 0} {
			p := SpinPol()
			jx := j[2] * p
			jy := j[1] * p
			jz := j[0] * p
			cuda.AddZhangLiTorque(dst, M.buffer, [3]float64{jx, jy, jz}, Msat(), nil, Alpha(), Xi())
		}
	})
	quants["sttorque"] = STTorque

	Torque = newSetter(3, global_mesh(), "torque", "T", func(b *data.Slice, good bool) {
		LLGTorque.set(b, good)
		STTorque.addTo(b, good)
	})
	quants["torque"] = Torque

	// solver
	torqueFn := func(good bool) *data.Slice {
		itime++
		Table.arm(good)    // if table output needed, quantities marked for update
		M.notifySave(good) // saves m if needed
		ExchangeMask.notifySave(good)

		Torque.set(torquebuffer, good)

		Table.touch(good) // all needed quantities are now up-to-date, save them
		return torquebuffer
	}
	Solver = cuda.NewHeun(M.buffer, torqueFn, cuda.Normalize, 1e-15, Gamma0, &Time)
}

// Register function f to be called after every time step.
// Typically used, e.g., to manipulate the magnetization.
func PostStep(f func()) {
	postStep = append(postStep, f)
}

func step() {
	Solver.Step()
	for _, f := range postStep {
		f()
	}
	s := Solver
	util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, Time, s.Dt_si, s.LastErr)
}

// injects arbitrary code into the engine run loops. Used by web interface.
var inject = make(chan func()) // inject function calls into the cuda main loop. Executed in between time steps.

// inject code into engine and wait for it to complete.
func injectAndWait(task func()) {
	ready := make(chan int)
	inject <- func() { task(); ready <- 1 }
	<-ready
}

// Returns the mesh cell size in meters. E.g.:
// 	cellsize_x := CellSize()[X]
func CellSize() [3]float64 {
	c := global_mesh().CellSize()
	return [3]float64{c[Z], c[Y], c[X]} // swaps XYZ
}

func WorldSize() [3]float64 {
	w := global_mesh().WorldSize()
	return [3]float64{w[Z], w[Y], w[X]} // swaps XYZ
}

func GridSize() [3]int {
	n := global_mesh().Size()
	return [3]int{n[Z], n[Y], n[X]} // swaps XYZ
}

func Nx() int { return GridSize()[X] }
func Ny() int { return GridSize()[Y] }
func Nz() int { return GridSize()[Z] }

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	log.Println("run for", seconds, "s")
	stop := Time + seconds
	RunCond(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	log.Println("run for", n, "steps")
	stop := Solver.NSteps + n
	RunCond(func() bool { return Solver.NSteps < stop })
}

// Runs as long as condition returns true.
func RunCond(condition func() bool) {
	checkInited() // todo: check in handler
	defer util.DashExit()

	pause = false
	for condition() && !pause {
		select {
		default:
			step()
		case f := <-inject:
			f()
		}
	}
	pause = true
}

// Enter interactive mode. Simulation is now exclusively controlled
// by web GUI (default: http://localhost:35367)
func RunInteractive() {
	lastKeepalive = time.Now()
	pause = true
	log.Println("entering interactive mode")
	if webPort == "" {
		goServe(*flag_port)
	}

	for {
		if time.Since(lastKeepalive) > webtimeout {
			log.Println("interactive session idle: exiting")
			break
		}
		log.Println("awaiting browser interaction")
		f := <-inject
		f()
	}
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	var zeromesh data.Mesh
	if globalmesh != zeromesh {
		free()
	}
	if Nx <= 1 {
		log.Fatal("mesh size X should be > 1, have: ", Nx)
	}
	globalmesh = *data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", global_mesh().UserString())
	initialize()
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	gridsize []int
	cellsize []float64
)

func setGridSize(nx, ny, nz float64) {
	Nx, Ny, Nz := cint(nx), cint(ny), cint(nz)
	gridsize = []int{Nx, Ny, Nz}
	if cellsize != nil {
		SetMesh(Nx, Ny, Nz, cellsize[0], cellsize[1], cellsize[2])
	}
}

func setCellSize(cx, cy, cz float64) {
	cellsize = []float64{cx, cy, cz}
	if gridsize != nil {
		SetMesh(gridsize[0], gridsize[1], gridsize[2], cx, cy, cz)
	}
}

// TODO: not perfectly OK yet.
func free() {
	log.Println("resetting gpu")
	cuda5.DeviceReset() // does not seem to clear allocations
	Init()
	dlQue = nil
}

// TODO: rename checkEningeInited or so
func checkInited() {
	if globalmesh.Size() == [3]int{0, 0, 0} {
		log.Fatal("need to set mesh first")
	}
}
