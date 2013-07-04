package engine

import ()

func init() {
	World.LValue("dmi", &DMI)
	DMI = scalarParam("dmi", "J/m2", func(r int) {
		dmi_red.setRegion(r, safediv(DMI.GetRegion(r), Msat.GetRegion(r)))
	})
}

var (
	DMI     ScalarParam                         // Dzyaloshinskii-Moriya strength in J/m²
	dmi_red = scalarParam("dmi_red", "Tm", nil) // DMI/Msat
)

func safediv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
