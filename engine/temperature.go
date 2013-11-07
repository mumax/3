package engine

import (
	"github.com/barnex/cuda5/curand"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

var (
	Temp      ScalarParam
	temp_red  derivedParam
	B_therm   adder
	E_therm   *GetScalar
	generator curand.Generator
	thermSeed int64 = 0
)

func init() {
	Temp.init("Temp", "K", "Temperature", []derived{&temp_red})
	DeclFunc("ThermSeed", ThermSeed, "Set a random seed for thermal noise")
	B_therm.init(3, &globalmesh, "B_therm", "T", "Thermal field", AddThermalField)
	E_therm = NewGetScalar("E_therm", "J", "Thermal energy", getThermalEnergy)
	registerEnergy(getThermalEnergy)

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

func AddThermalField(dst *data.Slice) {
	if Temp.isZero() {
		return
	}
	util.AssertMsg(solvertype == 1, "Temperature can only be used with Euler solver (solvertype 1)")
	if generator == 0 {
		generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		generator.SetSeed(thermSeed)
	}

	N := globalmesh.NCell()
	kmu0_VgammaDt := mag.Mu0 * mag.Kb / (mag.Gamma0 * cellVolume() * Solver.Dt_si)

	noise := cuda.Buffer(1, &globalmesh)
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	for i := 0; i < 3; i++ {
		generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.AddTemperature(dst.Comp(i), noise, temp_red.gpuLUT1(),
			kmu0_VgammaDt, regions.Gpu(), stream0)
	}
}

func getThermalEnergy() float64 {
	return -cellVolume() * dot(&M_full, &B_therm)
}

// Seeds the thermal noise generator
func ThermSeed(seed int) {
	thermSeed = int64(seed)
	if generator != 0 {
		generator.SetSeed(thermSeed)
	}
}
