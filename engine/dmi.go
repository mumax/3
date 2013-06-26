package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.LValue("dmi", &DMI)
	world.ROnly("B_dmi", &B_dmi)
	DMI = scalarParam("dmi", "J/m2", func(r int) {
		dmi_red.setRegion(r, safediv(DMI.GetRegion(r), Msat.GetRegion(r)))
	})
}

var (
	DMI     ScalarParam                         // Dzyaloshinskii-Moriya strength in J/m²
	dmi_red = scalarParam("dmi_red", "Tm", nil) // DMI/Msat
	B_dmi   adderQuant                          // DMI field in T
)

func initDMI() {
	B_dmi = adder(3, Mesh(), "B_dmi", "T", func(dst *data.Slice) {
		if !dmi_red.zero {
			cuda.AddDMI(dst, M.buffer, dmi_red.Gpu(), regions.Gpu())
		}
	})
	Quants["B_dmi"] = &B_dmi
}

func safediv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
