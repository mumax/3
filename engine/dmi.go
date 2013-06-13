package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.Var("dmi", &DMI)
	B_dmi_addr := &B_dmi
	world.ROnly("B_dmi", &B_dmi_addr)
}

var (
	DMI   func() float64 = Const(0) // Dzyaloshinskii-Moriya vector in J/m²
	B_dmi adderQuant                // DMI field in T
)

func initDMI() {
	B_dmi = adder(3, Mesh(), "B_dmi", "T", func(dst *data.Slice) {
		d := DMI()
		if d != 0 {
			cuda.AddDMI(dst, M.buffer, d, Msat.GetUniform())
		}
	})
	Quants["B_dmi"] = &B_dmi
}
