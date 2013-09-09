package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Anisotropy variables
var (
	Ku1, Kc1              ScalarParam
	AnisU, AnisC1, AnisC2 VectorParam
	ku1_red, kc1_red      param
	B_anis                adder // field due to uniaxial anisotropy (T)
	E_anis                = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", getAnisotropyEnergy)
)

func init() {
	Ku1.init("Ku1", "J/m3", "Uniaxial anisotropy constant")
	Kc1.init("Kc1", "J/m3", "Cubic anisotropy constant")
	AnisU.init("anisU", "", "Uniaxial anisotropy direction")
	AnisC1.init("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2.init("anisC2", "", "Cubic anisotorpy directon #2")

	ku1_red.init_(1, "ku1_red", "T", func() {
		if Ku1.timestamp() != Time || Msat.timestamp() != Time {
			paramDiv(&ku1_red, Ku1.Cpu(), Msat.Cpu())
		}
	})

	kc1_red.init_(1, "kc1_red", "T", func() {
		if Kc1.timestamp() != Time || Msat.timestamp() != Time {
			paramDiv(&ku1_red, Kc1.Cpu(), Msat.Cpu())
		}
	})

	B_anis.init(3, &globalmesh, "B_anis", "T", "Anisotropy field", func(dst *data.Slice) {
		if !ku1_red.zero() {
			cuda.AddUniaxialAnisotropy(dst, M.buffer, ku1_red.Gpu1(), AnisU.Gpu(), regions.Gpu())
		}
		if !kc1_red.zero() {
			cuda.AddCubicAnisotropy(dst, M.buffer, kc1_red.Gpu1(), AnisC1.Gpu(), AnisC2.Gpu(), regions.Gpu())
		}
	})

	registerEnergy(getAnisotropyEnergy)
}

func getAnisotropyEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_anis)
}

// dst = a/b, unless b == 0
func paramDiv(dst *param, a, b [][NREGION]float32) {
	util.Assert(dst.NComp() == 1 && len(a) == 1 && len(b) == 1)

	dst.gpu_ok = false
	for i := 0; i < regions.maxreg; i++ {
		a := a[0][i]
		b := b[0][i]
		if b == 0 {
			dst.cpu_buf[0][i] = 0
		} else {
			dst.cpu_buf[0][i] = a / b
		}
	}
}
