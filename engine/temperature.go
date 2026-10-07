package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

var (
	Temp        = NewScalarParam("Temp", "K", "Temperature")
	E_therm     = NewScalarValue("E_therm", "J", "Thermal energy", GetThermalEnergy)
	Edens_therm = NewScalarField("Edens_therm", "J/m3", "Thermal energy density", AddThermalEnergyDensity)
	B_therm     thermField // Thermal effective field (T)
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
	DeclFunc("ThermSeed", ThermSeed, "Deprecated: use ThermReseed() instead.<br>ThermSeed() sets a random seed for thermal noise, but does not reset the RNG offset, yielding different noise even for the same seed. This function remains here to ensure backwards compatibility with mumax3.12 and earlier.")
	DeclFunc("ThermReseed", ThermReseed, "Set a random seed for thermal noise")
	registerEnergy(GetThermalEnergy, AddThermalEnergyDensity)
	B_therm.step = -1 // invalidate noise cache
	DeclROnly("B_therm", &B_therm, "Thermal field (T)")
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
	if NSteps == b.step && Dt_si == b.dt {
		return
	}

	// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers
	if NSteps == b.step && Dt_si != b.dt {
		for c := 0; c < 3; c++ {
			cuda.Madd2(b.noise.Comp(c), b.noise.Comp(c), b.noise.Comp(c), float32(math.Sqrt(b.dt/Dt_si)), 0.)
		}
		b.dt = Dt_si
		return
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()

	// CURAND's GenerateNormal uses Box-Muller, which requires an even length. If N is odd, we allocate
	// one extra float so CURAND is satisfied. The settemperature2 kernel guards
	// on i < N (derived from B's size, not noise's size), so noise[N] is written but never read.
	Npad := int64(N + (N % 2))
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)

	// Use the exact mesh shape when N is already even (common case, no overhead), equivalent to Mesh().Size()
	// When N is odd, use a flat [Npad,1,1] shape — the Cuda kernel only cares about
	// the underlying pointer and the explicit N count, not the declared shape.
	noise := cuda.Buffer(1, [3]int{int(Npad), 1, 1})

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
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), Npad, mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt, ms, temp, alpha)
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
	warnStr := " WARNING: ThermSeed() is deprecated. Use ThermReseed() instead.\n" + // First "//" gets added by LogOut()
		"//          (ThermSeed does not guarantee identical noise after each call\n" +
		"//           because it only resets the RNG seed, not the RNG offset.)"
	LogOut(warnStr)
	B_therm.seed = int64(seed)
	if B_therm.generator != 0 {
		B_therm.generator.SetSeed(B_therm.seed)
	}
}

func ThermReseed(seed int) {
	B_therm.seed = int64(seed)
	if B_therm.generator != 0 {
		B_therm.generator.SetOffset(0)
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
