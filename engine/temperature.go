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
	E_therm   = NewGetScalar("E_therm", "J", "Thermal energy", getThermalEnergy)
	generator curand.Generator
)

func init() {
	Temp.init("Temp", "K", "Temperature", []derived{&temp_red})

	// TODO: derived parameters are a bit fragile
	temp_red.init(1, []updater{&Alpha, &Temp, &Msat}, func(p *derivedParam) {
		dst := temp_red.cpu_buf
		alpha := Alpha.cpuLUT()
		T := Temp.cpuLUT()
		Ms := Msat.cpuLUT()
		for i := 0; i < NREGION; i++ { // not regions.MaxReg!
			dst[0][i] = safediv(alpha[0][i]*T[0][i], mag.Mu0*Ms[0][i])
		}
	})

	B_therm.init(3, &globalmesh, "B_therm", "T", "Thermal field", func(dst *data.Slice) {
		if !Temp.isZero() {
			util.AssertMsg(solvertype == 1, "Temperature can only be used with Euler solver (solvertype 1)")
			if generator == 0 {
				generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
				generator.SetSeed(0) // TODO
			}

			N := globalmesh.NCell()
			kmu0_VgammaDt := mag.Mu0 * mag.Kb / (mag.Gamma0 * cellVolume() * Solver.Dt_si)

			noise := cuda.Buffer(1, &globalmesh)
			defer cuda.Recycle(noise)

			for i := 0; i < 3; i++ {
				generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), 0, 1)
				cuda.AddTemperature(dst.Comp(i), noise, temp_red.gpuLUT1(),
					kmu0_VgammaDt, regions.Gpu())
			}
		}
	})

	registerEnergy(getThermalEnergy)
}

func getThermalEnergy() float64 {
	return -cellVolume() * dot(&M_full, &B_therm)
}
