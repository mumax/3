package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Ku1     ScalarParam // Uniaxial anisotropy strength (J/m³)
	Kc1     ScalarParam // Cubic anisotropy strength (J/m³)
	AnisU   = vectorParam("anisU", "", nil)
	AnisC1  = vectorParam("anisC1", "", nil)
	AnisC2  = vectorParam("anisC2", "", nil)
	ku1_red = scalarParam("ku1_red", "T", nil) // Ku1 / Msat (T), auto updated from Ku1
	kc1_red = scalarParam("kc1_red", "T", nil) // Kc1 / Msat (T), auto updated from Kc1
	B_anis  adder                              // field due to uniaxial anisotropy (T)
	E_anis  = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", getAnisotropyEnergy)
)

func init() {
	Ku1 = scalarParam("Ku1", "J/m3", func(region int) {
		ku1_red.setRegion(region, safediv(Ku1.GetRegion(region), Msat.GetRegion(region)))
	})
	Kc1 = scalarParam("Kc1", "J/m3", func(region int) {
		kc1_red.setRegion(region, safediv(Kc1.GetRegion(region), Msat.GetRegion(region)))
	})
	DeclLValue("Ku1", &Ku1, "Uniaxial anisotropy constant (J/m³)")
	DeclLValue("Kc1", &Kc1, "Cubic anisotropy constant (J/m³)")
	DeclLValue("AnisU", &AnisU, "Uniaxial anisotropy direction")
	DeclLValue("AnisC1", &AnisC1, "Cubic anisotropy direction #1")
	DeclLValue("AnisC2", &AnisC2, "Cubic anisotorpy directon #2")

	B_anis.init(3, &globalmesh, "B_anis", "T", "Anisotropy field", func(dst *data.Slice) {
		if !ku1_red.zero() {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, ku1_red.Gpu(), AnisU.Gpu(), regions.Gpu())
		}
		if !kc1_red.zero() {
			cuda.AddCubicAnisotropy(dst, M.buffer, kc1_red.Gpu(), AnisC1.Gpu(), AnisC2.Gpu(), regions.Gpu())
		}
	})

	registerEnergy(getAnisotropyEnergy)

}

func getAnisotropyEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_anis)
}
