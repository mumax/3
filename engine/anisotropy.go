package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	AnisU  VectorParam // Uniaxial anisotropy axis
	Ku1    ScalarParam // Uniaxial anisotropy strength (J/m³)
	Kc1    ScalarParam // Cubic anisotropy strength (J/m³)
	AnisC1 VectorParam // Cubic anisotropy axis 1
	AnisC2 VectorParam // Cubic anisotropy axis 2
	B_anis adderQuant  // field due to uniaxial anisotropy (T)
	E_anis GetFunc     // Total anisotropy energy (J)

	ku1_red = scalarParam("ku1_red", "T", nil) // Ku1 / Msat (T), auto updated from Ku1 (TODO: form msat)
	kc1_red = scalarParam("kc1_red", "T", nil) // Kc1 / Msat (T), auto updated from Kc1 (TODO: form msat)
)

func init() {
	AnisU = vectorParam("anisU", "", nil)
	AnisC1 = vectorParam("anisC1", "", nil)
	AnisC2 = vectorParam("anisC2", "", nil)
	Ku1 = scalarParam("Ku1", "J/m3", func(region int) {
		ku1_red.setRegion(region, safediv(Ku1.GetRegion(region), Msat.GetRegion(region)))
	})
	Kc1 = scalarParam("Kc1", "J/m3", func(region int) {
		kc1_red.setRegion(region, safediv(Kc1.GetRegion(region), Msat.GetRegion(region)))
	})
	E_anis = NewGetScalar("E_anis", "J", getAnisotropyEnergy) // TODO: verify
	World.LValue("AnisU", &AnisU, "Uniaxial anisotropy direction")
	World.LValue("AnisC1", &AnisC1, "Cubic anisotropy direction #1")
	World.LValue("AnisC2", &AnisC2, "Cubic anisotorpy directon #2")
	World.LValue("Ku1", &Ku1, "Uniaxial anisotropy constant (J/m³)")
	World.LValue("Kc1", &Kc1, "Cubic anisotropy constant (J/m³)")
	World.ROnly("B_anis", &B_anis, "Anisotropy field (T)")
	World.ROnly("E_anis", &E_anis, "Anisotorpy energy (J)")
}

func initAnisotropy() {
	B_anis = adder(3, Mesh(), "B_anis", "T", func(dst *data.Slice) {
		if !ku1_red.zero {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, ku1_red.Gpu(), AnisU.Gpu(), regions.Gpu())
		}
		if !kc1_red.zero {
			cuda.AddCubicAnisotropy(dst, M.buffer, kc1_red.Gpu(), AnisC1.Gpu(), AnisC2.Gpu(), regions.Gpu())
		}
	})
	registerEnergy(getAnisotropyEnergy)
	Quants["B_anis"] = &B_anis
	Quants["Ku1"] = &Ku1
	Quants["Kc1"] = &Kc1
	Quants["anisU"] = &AnisU
	Quants["anisC1"] = &AnisC1
	Quants["anisC2"] = &AnisC2
}

func getAnisotropyEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_anis) / Mu0
}
