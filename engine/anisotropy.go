package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

// Anisotropy variables
var (
	Ku1, Kc1              ScalarParam  // uniaxial and cubic anis constants
	AnisU, AnisC1, AnisC2 VectorParam  // unixial and cubic anis axes
	ku1_red, kc1_red      derivedParam // K1 / Msat
	B_anis                adder        // field due to uniaxial anisotropy (T)
	E_anis                *GetScalar   // Anisotorpy energy
	Edens_anis            adder        // Anisotropy energy density
)

func init() {
	Ku1.init("Ku1", "J/m3", "Uniaxial anisotropy constant", []derived{&ku1_red})
	Kc1.init("Kc1", "J/m3", "Cubic anisotropy constant", []derived{&kc1_red})
	AnisU.init("anisU", "", "Uniaxial anisotropy direction")
	AnisC1.init("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2.init("anisC2", "", "Cubic anisotorpy directon #2")
	B_anis.init(VECTOR, "B_anis", "T", "Anisotropy field", AddAnisotropyField)
	E_anis = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", getAnisotropyEnergy)
	Edens_anis.init(SCALAR, "Edens_anis", "J/m3", "Anisotropy energy density (uni+cubic)", AddAnisotropyEdens)
	registerEnergy(getAnisotropyEnergy, AddAnisotropyEdens)

	//ku1_red = Ku1 / Msat
	ku1_red.init(1, []updater{&Ku1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Ku1.cpuLUT(), Msat.cpuLUT())
	})

	//ku1_red = Ku1 / Msat
	kc1_red.init(SCALAR, []updater{&Kc1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc1.cpuLUT(), Msat.cpuLUT())
	})
}

// Add the anisotropy field to dst
func AddAnisotropyField(dst *data.Slice) {
	if !(ku1_red.isZero()) {
		cuda.AddUniaxialAnisotropy(dst, M.Buffer(), ku1_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
	}
	if !(kc1_red.isZero()) {
		cuda.AddCubicAnisotropy(dst, M.Buffer(), kc1_red.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
	}
}

func getAnisotropyEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_anis)
}

func AddAnisotropyEdens(dst *data.Slice) {
	B, r := B_anis.Slice()
	if r {
		defer cuda.Recycle(B)
	}
	prefac := float32(-0.5 * cellVolume() * mag.Mu0)
	cuda.AddDotProduct(dst, prefac, B, M.Buffer(), geometry.Gpu())
}

// dst = a/b, unless b == 0
func paramDiv(dst, a, b [][NREGION]float32) {
	util.Assert(len(dst) == 1 && len(a) == 1 && len(b) == 1)
	for i := 0; i < NREGION; i++ { // not regions.maxreg
		dst[0][i] = safediv(a[0][i], b[0][i])
	}
}
