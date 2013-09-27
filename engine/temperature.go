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
	B_therm   adder
	E_therm   = NewGetScalar("E_therm", "J", "Thermal energy", getThermalEnergy)
	generator curand.Generator
)

func init() {
	Temp.init("Temp", "K", "Temperature", nil)

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
				cuda.AddTemperature(dst.Comp(i), noise, Temp.gpuLUT1(), Alpha.gpuLUT1(), Bsat.gpuLUT1(),
					kmu0_VgammaDt, regions.Gpu())
			}
		}
	})

	registerEnergy(getThermalEnergy)
}

func getThermalEnergy() float64 {
	return -cellVolume() * dot(&M_full, &B_therm)
}
