package engine

// Magnetocrystalline anisotropy.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Anisotropy variables
var (
	Ku1, Kc1, Kc2         ScalarParam  // uniaxial and cubic anis constants
	AnisU, AnisC1, AnisC2 VectorParam  // unixial and cubic anis axes
	ku1_red               derivedParam // K1 / Msat
	kc1_red, kc2_red      derivedParam
	B_anis                vAdder     // field due to uniaxial anisotropy (T)
	E_anis                *GetScalar // Anisotorpy energy
	Edens_anis            sAdder     // Anisotropy energy density
)

var zero ScalarParam // utility zero parameter

func init() {
	Ku1.init("Ku1", "J/m3", "Uniaxial anisotropy constant", []derived{&ku1_red})
	Kc1.init("Kc1", "J/m3", "1st order cubic anisotropy constant", []derived{&kc1_red})
	Kc2.init("Kc2", "J/m3", "2nd order cubic anisotropy constant [UNTESTED]", []derived{&kc2_red})
	AnisU.init("anisU", "", "Uniaxial anisotropy direction")
	AnisC1.init("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2.init("anisC2", "", "Cubic anisotorpy directon #2")
	B_anis.init("B_anis", "T", "Anisotropy field", AddAnisotropyField)
	E_anis = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", GetAnisotropyEnergy)
	Edens_anis.init("Edens_anis", "J/m3", "Anisotropy energy density (uni+cubic)", AddAnisotropyEnergyDensity)
	registerEnergy(GetAnisotropyEnergy, Edens_anis.AddTo)

	//ku1_red = Ku1 / Msat
	ku1_red.init(1, []updater{&Ku1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Ku1.cpuLUT(), Msat.cpuLUT())
	})

	//kc1_red = Kc1 / Msat
	kc1_red.init(SCALAR, []updater{&Kc1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc1.cpuLUT(), Msat.cpuLUT())
	})

	//kc2_red = Kc2 / Msat
	kc2_red.init(SCALAR, []updater{&Kc2, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc2.cpuLUT(), Msat.cpuLUT())
	})
}

// Add the anisotropy field to dst
func AddAnisotropyField(dst *data.Slice) {
	if !(ku1_red.isZero()) {
		cuda.AddUniaxialAnisotropy(dst, M.Buffer(), ku1_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
	}
	if !(kc1_red.isZero()) || !(kc2_red.isZero()) {
		cuda.AddCubicAnisotropy(dst, M.Buffer(), kc1_red.gpuLUT1(), kc2_red.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
	}
}

func AddAnisotropyEnergyDensity(dst *data.Slice) {
	buf := cuda.Buffer(B_anis.NComp(), B_anis.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf, r := M_full.Slice()
	if r {
		defer cuda.Recycle(Mf)
	}

	if !Ku1.isZero() {
		cuda.Zero(buf)
		cuda.AddUniaxialAnisotropy(buf, M.Buffer(), ku1_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -0.5, buf, Mf)
	}

	if !Kc1.isZero() {
		cuda.Zero(buf)
		cuda.AddCubicAnisotropy(buf, M.Buffer(), kc1_red.gpuLUT1(), kc2_red.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -0.25, buf, Mf)
	}
}

// Returns anisotropy energy in joules.
func GetAnisotropyEnergy() float64 {
	buf := cuda.Buffer(B_anis.NComp(), B_anis.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf, r := M_full.Slice()
	if r {
		defer cuda.Recycle(Mf)
	}

	// add terms with correct prefactor
	E := 0.

	if !Ku1.isZero() {
		cuda.Zero(buf)
		cuda.AddUniaxialAnisotropy(buf, M.Buffer(), ku1_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
		E += -0.5 * cellVolume() * float64(cuda.Dot(Mf, buf))
	}

	if !Kc1.isZero() {
		cuda.Zero(buf)
		cuda.AddCubicAnisotropy(buf, M.Buffer(), kc1_red.gpuLUT1(), kc2_red.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
		E += -0.25 * cellVolume() * float64(cuda.Dot(Mf, buf))
	}

	return E
}
