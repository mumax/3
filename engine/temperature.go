package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Temp    ScalarParam
	B_therm adder
	E_therm = NewGetScalar("E_therm", "J", "Thermal energy", getThermalEnergy)
)

func init() {
	Temp.init("Temp", "K", "Temperature")

	B_therm.init(3, &globalmesh, "B_therm", "T", "Thermal field", func(dst *data.Slice) {
		if !(Temp.isZero()) {
			//cuda.AddTemperature(dst, M.buffer, ku1_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
		}
	})

	registerEnergy(getThermalEnergy)
}

func getThermalEnergy() float64 {
	return -cellVolume() * dot(&M_full, &B_therm)
}
