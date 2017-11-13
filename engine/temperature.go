package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
	"math"
)

var (
	Temp        = NewScalarParam("Temp", "K", "Temperature")
	TCurie      = NewScalarParam("TCurie", "K", "Curie Temperature")
	E_therm     = NewScalarValue("E_therm", "J", "Thermal energy", GetThermalEnergy)
	Edens_therm = NewScalarField("Edens_therm", "J/m3", "Thermal energy density", AddThermalEnergyDensity)
	B_therm     thermField // Thermal effective field (T)
	//Qext		= NewScalarParam("Qext", "W/m3", "External Heating")
	Qext		= NewExcitation("Qext", "W/m3", "External Heating")  // To do Derive a class ScalarExcitation
        // For Joule heating
	TempJH      LocalTemp   // We will only use .noise for local temperature [0] and temperature dependencies
	Kthermal    = NewScalarParam("Kthermal", "W/(m·K)", "Thermal conductivity")
	Cthermal    = NewScalarParam("Cthermal", "J/(Kg·K)", "Specific heat capacity")
	Resistivity = NewScalarParam("Resistivity", "Ohm·m", "Electric resistivity")
	Density     = NewScalarParam("Density", "Kg/m3", "Mass density")
	TSubs       = NewScalarParam("TSubs", "K", "Substrate Temperature")
	TauSubs     = NewScalarParam("TauSubs", "s", "Substrate difussion time")

)

var AddThermalEnergyDensity = makeEdensAdder(&B_therm, -1)

// thermField calculates and caches thermal noise.
type thermField struct {
	seed      int64            // seed for generator
	generator curand.Generator //
	noise     *data.Slice      // noise buffer
	step      int              // solver step corresponding to noise
	dt        float64          // solver timestep corresponding to noise
}

func init() {
	DeclFunc("ThermSeed", ThermSeed, "Set a random seed for thermal noise")
	registerEnergy(GetThermalEnergy, AddThermalEnergyDensity)
	B_therm.step = -1 // invalidate noise cache
	DeclROnly("B_therm", &B_therm, "Thermal field (T)")

        // For JH (see at the end)
	DeclROnly("TempJH", AsScalarField(&TempJH), "Local Temperature (K)")
	DeclFunc("RestartJH", StartJH, "Equals Temperature to substrate")
	DeclFunc("GetTemp", GetCell, "Gets cell temperature")
}


func (b *thermField) AddTo(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		cuda.Add(dst, dst, b.noise)
	}
}

func (b *thermField) update() {
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.

	if FixDt != 0 {
		Dt_si = FixDt
	}

	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		// when noise was (re-)allocated it's invalid for sure.
		B_therm.step = -1
		B_therm.dt = -1
	}

	if Temp.isZero() {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt && solvertype!=7 && solvertype!=8 {
		// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers		
		if NSteps == b.step && Dt_si != b.dt {		
		for c := 0; c < 3; c++ {		
			cuda.Madd2(b.noise.Comp(c), b.noise.Comp(c), b.noise.Comp(c), float32(math.Sqrt(b.dt/Dt_si)), 0.)		
		}		
		b.dt = Dt_si		
 		return
 		}
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	ms := Msat.MSlice()
	defer ms.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		if (solvertype!=8) {
                               cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt, ms, temp, alpha)
                               } else{
			       TempJH.update()
                               cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, ms, TempJH.temp, alpha)
                               }
	}

	b.step = NSteps
	b.dt = Dt_si
}

func GetThermalEnergy() float64 {
	if Temp.isZero() || relaxing {
		return 0
	} else {
		return -cellVolume() * dot(&M_full, &B_therm)
	}
}

// Seeds the thermal noise generator
func ThermSeed(seed int) {
	B_therm.seed = int64(seed)
	if B_therm.generator != 0 {
		B_therm.generator.SetSeed(B_therm.seed)
	}
}

func (b *thermField) Mesh() *data.Mesh       { return Mesh() }
func (b *thermField) NComp() int             { return 3 }
func (b *thermField) Name() string           { return "Thermal field" }
func (b *thermField) Unit() string           { return "T" }
func (b *thermField) average() []float64     { return qAverageUniverse(b) }
func (b *thermField) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *thermField) Slice() (*data.Slice, bool) {
	b.update()
	return b.noise, false
}


// LocalTemp definitions and Functions for JH

type LocalTemp struct {
	temp     *data.Slice      // noise buffer
}

func StartJH() {
	TempJH.JHSetLocalTemp()
}

func (b *LocalTemp) JHSetLocalTemp() {
//func (b *thermField) JHSetLocalTemp() {
	b.update()
	TSubs := TSubs.MSlice()
	defer TSubs.Recycle()
	cuda.InitTemperatureJH(b.temp,TSubs)
}

func (b *LocalTemp) update() {
	if b.temp == nil {
		b.temp = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		TSubs := TSubs.MSlice()
		defer TSubs.Recycle()
		cuda.InitTemperatureJH(b.temp,TSubs)
	}
}

func (b *LocalTemp) Mesh() *data.Mesh       { return Mesh() }
func (b *LocalTemp) NComp() int             { return 1 }
func (b *LocalTemp) Name() string           { return "LocalTemperature" }
func (b *LocalTemp) Unit() string           { return "K" }
func (b *LocalTemp) average() []float64     { return qAverageUniverse(b) }
func (b *LocalTemp) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *LocalTemp) Slice() (*data.Slice, bool) {
	b.update()
	return b.temp, false
}

func GetCell(ix, iy, iz int) float64 {
	return float64(TempJH.GetCell(ix, iy, iz))
}
func (b *LocalTemp) GetCell(ix, iy, iz int) float32 { 
	return cuda.GetCell(b.temp, 0, ix, iy, iz)}
