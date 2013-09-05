package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Ku1     = scalarParam("Ku1", "J/m3")
	Kc1     = scalarParam("Kc1", "J/m3")
	AnisU   = vectorParam("anisU", "", nil)
	AnisC1  = vectorParam("anisC1", "", nil)
	AnisC2  = vectorParam("anisC2", "", nil)
	ku1_red = scalarParam("ku1_red", "T", &Ku1, &Msat) // Ku1 / Msat (T)
	kc1_red = scalarParam("kc1_red", "T", &Kc1, &Msat) // Kc1 / Msat (T)
	B_anis  adder                                      // field due to uniaxial anisotropy (T)
	E_anis  = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", getAnisotropyEnergy)
)

func init() {

	ku1_red.updateAll = func() {
		safediv(ku1_red.cpu, Ku1.Cpu(), Msat.Cpu())
	}

	kc1_red.updateAll = func() {
		safediv(kc1_red.cpu, Kc1.Cpu(), Msat.Cpu())
	}

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
