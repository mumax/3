package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Anisotropy variables
var (
	Ku1, Kc1              ScalarParam  // uniaxial and cubic anis constants
	AnisU, AnisC1, AnisC2 VectorParam  // unixial and cubic anis axes
	ku1_red, kc1_red      derivedParam // K1 / Msat
	B_anis                adder        // field due to uniaxial anisotropy (T)
	E_anis                = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", getAnisotropyEnergy)
)

func init() {
	Ku1.init("Ku1", "J/m3", "Uniaxial anisotropy constant", []derived{&ku1_red})
	Kc1.init("Kc1", "J/m3", "Cubic anisotropy constant", []derived{&kc1_red})
	AnisU.init("anisU", "", "Uniaxial anisotropy direction")
	AnisC1.init("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2.init("anisC2", "", "Cubic anisotorpy directon #2")

	ku1_red.init(1, []updater{&Ku1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Ku1.CpuLUT(), Msat.CpuLUT())
	})

	kc1_red.init(1, []updater{&Kc1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc1.CpuLUT(), Msat.CpuLUT())
	})

	B_anis.init(3, &globalmesh, "B_anis", "T", "Anisotropy field", func(dst *data.Slice) {
		if !(ku1_red.isZero()) {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, ku1_red.LUT1(), AnisU.LUT(), regions.Gpu())
		}
		if !(kc1_red.isZero()) {
			cuda.AddCubicAnisotropy(dst, M.buffer, kc1_red.LUT1(), AnisC1.LUT(), AnisC2.LUT(), regions.Gpu())
		}
	})

	registerEnergy(getAnisotropyEnergy)
}

func getAnisotropyEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_anis)
}

// dst = a/b, unless b == 0
func paramDiv(dst, a, b [][NREGION]float32) {
	util.Assert(len(dst) == 1 && len(a) == 1 && len(b) == 1)
	for i := 0; i < NREGION; i++ { // todo: regions.maxreg?
		dst[0][i] = safediv(a[0][i], b[0][i])
	}
}
