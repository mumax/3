package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

var (
	Temp        ScalarParam  // Temperature in K
	temp_red    derivedParam // reduced temperature = (alpha * Temp) / (mu0 * Msat)
	E_therm     *GetScalar   // Thermal energy in J
	Edens_therm sAdder       // Thermal energy density
	B_therm     thermField   // Thermal effective field (T)
)

// thermField calculates and caches thermal noise.
type thermField struct {
	seed      int64            // seed for generator
	generator curand.Generator //
	noise     *data.Slice      // noise buffer
	step      int              // solver step corresponding to noise
	dt        float64          // solver timestep corresponding to noise
}

func init() {
	Temp.init("Temp", "K", "Temperature", []derived{&temp_red})
	DeclFunc("ThermSeed", ThermSeed, "Set a random seed for thermal noise")
	E_therm = NewGetScalar("E_therm", "J", "Thermal energy", GetThermalEnergy)
	Edens_therm.init("Edens_therm", "J/m3", "Thermal energy density", makeEdensAdder(&B_therm, -1))
	registerEnergy(GetThermalEnergy, Edens_therm.AddTo)
	B_therm.step = -1 // invalidate noise cache
	DeclROnly("B_therm", &B_therm, "Thermal field (T)")

	// reduced temperature = (alpha * T) / (mu0 * Msat)
	temp_red.init(1, []updater{&Alpha, &Temp, &Msat}, func(p *derivedParam) {
		dst := temp_red.cpu_buf
		alpha := Alpha.cpuLUT()
		T := Temp.cpuLUT()
		Ms := Msat.cpuLUT()
		for i := 0; i < NREGION; i++ { // not regions.MaxReg!
			dst[0][i] = safediv(alpha[0][i]*T[0][i], mag.Mu0*Ms[0][i])
		}
	})
}

func (b *thermField) AddTo(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		cuda.Madd2(dst, dst, b.noise, 1, 1)
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
	if NSteps == b.step && Dt_si == b.dt {
		return
	}

	if FixDt == 0 {
		util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2mu0_VgammaDt := 2 * mag.Mu0 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, temp_red.gpuLUT1(), k2mu0_VgammaDt, regions.Gpu())
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

func (b *thermField) Mesh() *data.Mesh   { return Mesh() }
func (b *thermField) NComp() int         { return 3 }
func (b *thermField) Name() string       { return "Thermal field" }
func (b *thermField) Unit() string       { return "T" }
func (b *thermField) average() []float64 { return qAverageUniverse(b) }
func (b *thermField) Slice() (*data.Slice, bool) {
	b.update()
	return b.noise, false
}
