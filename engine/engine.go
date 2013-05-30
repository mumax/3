package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	cuda5 "github.com/barnex/cuda5/cuda"
	"log"
)

// User inputs
var (
	Aex          func() float64     = Const(0)             // Exchange stiffness in J/m
	ExchangeMask staggeredMaskQuant                        // Mask that scales Aex/Msat between cells.
	KuMask       maskQuant                                 // Mask to scales Ku1/Msat cellwise.
	Msat         func() float64     = Const(0)             // Saturation magnetization in A/m
	Alpha        func() float64     = Const(0)             // Damping constant
	B_ext        func() [3]float64  = ConstVector(0, 0, 0) // Externally applied field in T, homogeneous.
	DMI          func() float64     = Const(0)             // Dzyaloshinskii-Moriya vector in J/m²
	Ku1          func() [3]float64  = ConstVector(0, 0, 0) // Uniaxial anisotropy vector in J/m³
	Xi           func() float64     = Const(0)             // Non-adiabaticity of spin-transfer-torque
	SpinPol      func() float64     = Const(1)             // Spin polarization of electrical current
	J            func() [3]float64  = ConstVector(0, 0, 0) // Electrical current density
	EnableDemag  bool               = true                 // enable/disable demag field
)

// Accessible quantities
var (
	M                 magnetization // reduced magnetization (unit length)
	FFTM              fftm          // FFT of M
	B_eff             setterQuant   // effective field (T) output handle
	B_demag           setterQuant   // demag field (T) output handle
	B_dmi             adderQuant    // demag field (T) output handle
	B_exch            adderQuant    // exchange field (T) output handle
	B_uni             adderQuant    // field due to uniaxial anisotropy output handle
	STTorque          adderQuant    // spin-transfer torque output handle
	LLGTorque, Torque setterQuant   // torque/gamma0, in Tesla
	Table             DataTable     // output handle for tabular data (average magnetization etc.)
	Time              float64       // time in seconds  // todo: hide? setting breaks autosaves
	Solver            cuda.Heun
	Geom              func(x, y, z float64) bool = func(x, y, z float64) bool { return true } // geometric stencil
)

// hidden quantities
var (
	globalmesh   data.Mesh
	torquebuffer *data.Slice
	vol          *data.Slice
	postStep     []func() // called on after every time step
	extFields    []extField
	itime        int //unique integer time stamp
	demag_       *cuda.DemagConvolution
)

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh
}

// Add an additional space-dependent field to B_ext.
// The field is mask * multiplier, where mask typically contains space-dependent scaling values of the order of 1.
// multiplier can be time dependent.
// TODO: extend API (set one component, construct masks or read from file). Also for current.
func AddExtField(mask *data.Slice, multiplier func() float64) {
	m := cuda.GPUCopy(mask)
	extFields = append(extFields, extField{m, multiplier})
}

type extField struct {
	mask *data.Slice
	mul  func() float64
}

// maps quantity names to downloadable data. E.g. for rendering
var Quants = make(map[string]Quant)

func initialize() {

	// these 2 GPU arrays are re-used to stored various quantities.
	torquebuffer = cuda.NewSlice(3, Mesh())

	// cell volumes currently unused
	vol = data.NilSlice(1, Mesh())

	// magnetization
	M.init()
	Quants["m"] = &M
	//AvgM = newScalar(3, "m", "", func() []float64 {
	//	return average(&M)
	//})

	FFTM.init()
	Quants["mFFT"] = &fftmPower{} // for the web interface we display FFT amplitude

	// data table
	Table = *newTable("datatable")

	// demag field
	demag_ = cuda.NewDemag(Mesh())
	B_demag = *newSetter(3, Mesh(), "B_demag", "T", func(b *data.Slice, cansave bool) {
		if EnableDemag {
			demag_.Exec(b, M.buffer, vol, Mu0*Msat())
		} else {
			cuda.Zero(b)
		}
	})
	Quants["B_demag"] = &B_demag

	// exchange field
	B_exch = *newAdder(3, Mesh(), "B_exch", "T", func(dst *data.Slice) {
		cuda.AddExchange(dst, M.buffer, ExchangeMask.buffer, Aex(), Msat())
	})
	Quants["B_exch"] = &B_exch

	ExchangeMask = *newStaggeredMask(Mesh(), "exchangemask", "")
	Quants["exchangemask"] = &ExchangeMask

	// Dzyaloshinskii-Moriya field
	B_dmi = *newAdder(3, Mesh(), "B_dmi", "T", func(dst *data.Slice) {
		d := DMI()
		if d != 0 {
			cuda.AddDMI(dst, M.buffer, d, Msat())
		}
	})
	Quants["B_dmi"] = &B_dmi

	// uniaxial anisotropy
	B_uni = *newAdder(3, Mesh(), "B_uni", "T", func(dst *data.Slice) {
		ku1 := Ku1() // in J/m3
		if ku1 != [3]float64{0, 0, 0} {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, KuMask.buffer, ku1[2], ku1[1], ku1[0], Msat())
		}
	})
	Quants["B_uni"] = &B_uni

	KuMask = *newMask(3, Mesh(), "kumask", "")
	Quants["KuMask"] = &KuMask

	// external field
	b_ext := newAdder(3, Mesh(), "B_ext", "T", func(dst *data.Slice) {
		bext := B_ext()
		cuda.AddConst(dst, float32(bext[2]), float32(bext[1]), float32(bext[0]))
		for _, f := range extFields {
			cuda.Madd2(dst, dst, f.mask, 1, float32(f.mul()))
		}
	})
	//Quants["B_ext"] = B_ext

	// effective field
	B_eff = *newSetter(3, Mesh(), "B_eff", "T", func(dst *data.Slice, cansave bool) {
		B_demag.set(dst, cansave)
		B_exch.addTo(dst, cansave)
		B_dmi.addTo(dst, cansave)
		B_uni.addTo(dst, cansave)
		b_ext.addTo(dst, cansave)
	})
	Quants["B_eff"] = &B_eff

	// llg torque
	LLGTorque = *newSetter(3, Mesh(), "llgtorque", "T", func(b *data.Slice, cansave bool) {
		B_eff.set(b, cansave)
		cuda.LLGTorque(b, M.buffer, b, float32(Alpha()))
	})
	Quants["llgtorque"] = &LLGTorque

	// spin-transfer torque
	STTorque = *newAdder(3, Mesh(), "sttorque", "T", func(dst *data.Slice) {
		j := J()
		if j != [3]float64{0, 0, 0} {
			p := SpinPol()
			jx := j[2] * p
			jy := j[1] * p
			jz := j[0] * p
			cuda.AddZhangLiTorque(dst, M.buffer, [3]float64{jx, jy, jz}, Msat(), nil, Alpha(), Xi())
		}
	})
	Quants["sttorque"] = &STTorque

	Torque = *newSetter(3, Mesh(), "torque", "T", func(b *data.Slice, cansave bool) {
		LLGTorque.set(b, cansave)
		STTorque.addTo(b, cansave)
	})
	Quants["torque"] = &Torque

	// solver
	torqueFn := func(cansave bool) *data.Slice {
		itime++
		Table.arm(cansave)    // if table output needed, quantities marked for update
		M.notifySave(cansave) // saves m if needed
		FFTM.notifySave(cansave)
		ExchangeMask.notifySave(cansave)

		Torque.set(torquebuffer, cansave)

		Table.touch(cansave) // all needed quantities are now up-to-date, save them
		return torquebuffer
	}
	Solver = *cuda.NewHeun(M.buffer, torqueFn, cuda.Normalize, 1e-15, Gamma0, &Time)
}

// Register function f to be called after every time step.
// Typically used, e.g., to manipulate the magnetization.
func PostStep(f func()) {
	postStep = append(postStep, f)
}

func init() {
	world.Func("PostStep", PostStep)
}

func step() {
	Solver.Step()
	for _, f := range postStep {
		f()
	}
	s := Solver
	util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, Time, s.Dt_si, s.LastErr)
}

// Returns the mesh cell size in meters. E.g.:
// 	cellsize_x := CellSize()[X]
func CellSize() [3]float64 {
	c := Mesh().CellSize()
	return [3]float64{c[Z], c[Y], c[X]} // swaps XYZ
}

func WorldSize() [3]float64 {
	w := Mesh().WorldSize()
	return [3]float64{w[Z], w[Y], w[X]} // swaps XYZ
}

func GridSize() [3]int {
	n := Mesh().Size()
	return [3]int{n[Z], n[Y], n[X]} // swaps XYZ
}

func Nx() int { return GridSize()[X] }
func Ny() int { return GridSize()[Y] }
func Nz() int { return GridSize()[Z] }

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
	log.Println("set mesh:", Mesh().UserString())
	initialize()
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	gridsize []int
	cellsize []float64
)

func setGridSize(Nx, Ny, Nz int) {
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

// check if mesh is set
func checkMesh() {
	if globalmesh.Size() == [3]int{0, 0, 0} {
		log.Fatal("need to set mesh first")
	}
}

// check if m is set
func checkM() {
	checkMesh()
	if M.buffer.DevPtr(0) == nil {
		log.Fatal("need to initialize magnetization first")
	}
	if cuda.MaxVecNorm(M.buffer) == 0 {
		log.Fatal("need to initialize magnetization first")
	}
}
