package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
)

// User inputs
var (
	Aex   ScalFn                        // Exchange stiffness in J/m
	Msat  ScalFn                        // Saturation magnetization in A/m
	Alpha ScalFn                        // Damping constant
	B_ext VecFn  = ConstVector(0, 0, 0) // External field in T
	DMI   ScalFn = Const(0)             // Dzyaloshinskii-Moriya vector in J/m²
	Ku1   VecFn  = ConstVector(0, 0, 0) // Uniaxial anisotropy vector in J/m³
)

// Accessible quantities
var (
	M       Settable // reduced magnetization output handle
	B_eff   Handle   // effective field (T) output handle
	Torque  Buffered // torque (?) output handle
	B_demag Handle   // demag field (T) output handle
	B_dmi   Handle   // demag field (T) output handle
	B_exch  Handle   // exchange field (T) output handle
	B_uni   Handle   // field due to uniaxial anisotropy output handle
	Table   Handle   // output handle for tabular data (average magnetization etc.)
	Time    float64  // time in seconds  // todo: hide? setting breaks autosaves
	Solver  *cuda.Heun
)

// hidden quantities
var (
	mesh *data.Mesh
	m    *buffered
	vol  *data.Slice
)

func initialize() {

	// these 2 GPU arrays are re-used to stored various quantities.
	arr1, arr2 := cuda.NewSynced(3, mesh), cuda.NewSynced(3, mesh)

	// cell volumes currently unused
	vol = data.NilSlice(1, mesh)

	// magnetization
	m = newBuffered(arr1, "m", nil)
	M = m

	// effective field
	b_eff := newBuffered(arr2, "B_eff", nil)
	B_eff = b_eff

	// demag field
	demag_ := cuda.NewDemag(mesh)
	b_demag := newBuffered(arr2, "B_demag", func(b *data.Slice) {
		m_ := m.Read()
		demag_.Exec(b, m_, vol, Mu0*Msat()) //TODO: consistent msat or bsat
		m.ReadDone()
	})
	B_demag = b_demag

	// exchange field
	b_exch := newAdder("B_exch", func(dst *data.Slice) {
		m_ := m.Read()
		cuda.AddExchange(dst, m_, Aex(), Msat())
		m.ReadDone()
	})
	B_exch = b_exch

	// Dzyaloshinskii-Moriya field
	b_dmi := newAdder("B_dmi", func(dst *data.Slice) {
		d := DMI()
		if d != 0 {
			m_ := m.Read()
			cuda.AddDMI(dst, m_, d, Msat())
			m.ReadDone()
		}
	})
	B_dmi = b_dmi

	// uniaxial anisotropy
	b_uni := newAdder("B_uni", func(dst *data.Slice) {
		ku1 := Ku1() // in J/m3
		if ku1 != [3]float64{0, 0, 0} {
			m_ := m.Read()
			cuda.AddUniaxialAnisotropy(dst, m_, ku1[2], ku1[1], ku1[0], Msat())
			m.ReadDone()
		}
	})
	B_uni = b_uni

	// external field
	b_ext := newAdder("B_ext", func(dst *data.Slice) {
		bext := B_ext()
		cuda.AddConst(dst, float32(bext[2]), float32(bext[1]), float32(bext[0]))
	})

	// llg torque
	torque := newBuffered(arr2, "torque", func(b *data.Slice) {
		m_ := m.Read()
		cuda.LLGTorque(b, m_, b, float32(Alpha()))
		m.ReadDone()
	})
	Torque = torque

	// data table
	table := newTable("datatable")
	Table = table

	// solver
	torqueFn := func(good bool) *data.Synced {
		m.touch(good) // saves if needed
		table.send(m.Synced, good)
		b_demag.update(good)
		b_exch.addTo(b_eff, good)
		b_dmi.addTo(b_eff, good)
		b_uni.addTo(b_eff, good)
		b_ext.addTo(b_eff, good)
		torque.update(good)
		return torque.Synced
	}
	Solver = cuda.NewHeun(m.Synced, torqueFn, cuda.Normalize, 1e-15, Gamma0, &Time)
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	log.Println("run for", seconds, "s")
	checkInited()
	stop := Time + seconds
	defer util.DashExit()
	for Time < stop {
		step()
	}
}

// Run the simulation for a number of steps.
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
	util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, Time, s.Dt_si, s.LastErr)
}

// Set the magnetization to uniform state. // TODO: mv to settable
func SetMUniform(mx, my, mz float32) {
	checkInited()
	m.memset(mz, my, mx)
	m.normalize()
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if mesh != nil {
		log.Fatal("mesh already set")
	}
	if Nx <= 1 {
		log.Fatal("mesh size X should be > 1, have: ", Nx)
	}
	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", mesh.UserString())
	initialize()
}

func GetMesh() *data.Mesh {
	return mesh
}

func checkInited() {
	if mesh == nil {
		log.Fatal("need to set mesh first")
	}
}

// map of names to Handle does not work because Handles change on the fly
// *Handle does not work because we loose interfaceness.
func Quant(name string) (h Buffered, ok bool) {
	switch name {
	default:
		return nil, false
	case "m":
		return M, true
	case "torque":
		return Torque, true
	}
}
