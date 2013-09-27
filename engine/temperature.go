package engine

import (
	"github.com/barnex/cuda5/curand"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
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
		if !(Temp.isZero()) {
			if generator == 0 {
				generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
				generator.SetSeed(0) // TODO
			}

			VolDt := cellVolume() * Solver.Dt_si
			noise := cuda.Buffer(1, &globalmesh)
			N := globalmesh.NCell()
			for i := 0; i < 3; i++ {
				generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), 0, 1)
				cuda.AddTemperature(dst, noise, Temp.gpuLUT1(), Alpha.gpuLUT1(), Bsat.gpuLUT1(), VolDt, regions.Gpu())
			}
		}
	})

	registerEnergy(getThermalEnergy)
}

func getThermalEnergy() float64 {
	return -cellVolume() * dot(&M_full, &B_therm)
}
