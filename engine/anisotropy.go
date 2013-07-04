package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	AnisU   = vectorParam("anisU", "", nil)    // Uniaxial anisotropy axis
	ku1_red = scalarParam("ku1_red", "T", nil) // Ku1 / Msat (T), auto updated from Ku1 (TODO: form msat)
	Ku1     ScalarParam                        // Uniaxial anisotropy strength (J/m³)
	B_anis  adderQuant                         // field due to uniaxial anisotropy output handle
	E_anis  = NewGetScalar("E_anis", "J", GetAnisotropyEnergy)
)

func init() {
	Ku1 = scalarParam("Ku1", "J/m3", func(region int) {
		ku1_red.setRegion(region, safediv(Ku1.GetRegion(region), Msat.GetRegion(region)))
	})
	World.LValue("AnisU", &AnisU)
	World.LValue("Ku1", &Ku1)
	World.ROnly("B_anis", &B_anis)
	World.ROnly("E_anis", &E_anis)
}

func initAnisotropy() {
	B_anis = adder(3, Mesh(), "B_anis", "T", func(dst *data.Slice) {
		if !ku1_red.zero {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, ku1_red.Gpu(), AnisU.Gpu(), regions.Gpu())
		}
	})
	registerEnergy(GetAnisotropyEnergy)
	Quants["B_anis"] = &B_anis
	Quants["Ku1"] = &Ku1
	Quants["anisU"] = &AnisU
}

func GetAnisotropyEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_anis) / Mu0
}
