package engine

// Magnetocrystalline anisotropy.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Anisotropy variables
var (
	Ku1, Ku2                  ScalarParam  // uniaxial anis constants
	Kc1, Kc2, Kc3             ScalarParam  // cubic anis constants
	AnisU, AnisC1, AnisC2     VectorParam  // unixial and cubic anis axes
	ku1_red, ku2_red          derivedParam // K1 / Msat
	kc1_red, kc2_red, kc3_red derivedParam
	B_anis                    vAdder     // field due to uniaxial anisotropy (T)
	E_anis                    *GetScalar // Anisotorpy energy
	Edens_anis                sAdder     // Anisotropy energy density
	zero                      inputParam // utility zero parameter
)

func init() {
	Ku1.init("Ku1", "J/m3", "1st order uniaxial anisotropy constant", []derived{&ku1_red})
	Ku2.init("Ku2", "J/m3", "2nd order uniaxial anisotropy constant", []derived{&ku2_red})
	Kc1.init("Kc1", "J/m3", "1st order cubic anisotropy constant", []derived{&kc1_red})
	Kc2.init("Kc2", "J/m3", "2nd order cubic anisotropy constant", []derived{&kc2_red})
	Kc3.init("Kc3", "J/m3", "3rd order cubic anisotropy constant", []derived{&kc3_red})
	AnisU.init("anisU", "", "Uniaxial anisotropy direction")
	AnisC1.init("anisC1", "", "Cubic anisotropy direction #1")
	AnisC2.init("anisC2", "", "Cubic anisotorpy directon #2")
	B_anis.init("B_anis", "T", "Anisotropy field", AddAnisotropyField)
	E_anis = NewGetScalar("E_anis", "J", "Anisotropy energy (uni+cubic)", GetAnisotropyEnergy)
	Edens_anis.init("Edens_anis", "J/m3", "Anisotropy energy density (uni+cubic)", AddAnisotropyEnergyDensity)
	registerEnergy(GetAnisotropyEnergy, Edens_anis.AddTo)
	zero.init(1, "_zero", "", nil)

	//ku1_red = Ku1 / Msat
	ku1_red.init(SCALAR, []updater{&Ku1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Ku1.cpuLUT(), Msat.cpuLUT())
	})
	//ku2_red = Ku2 / Msat
	ku2_red.init(SCALAR, []updater{&Ku2, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Ku2.cpuLUT(), Msat.cpuLUT())
	})

	//kc1_red = Kc1 / Msat
	kc1_red.init(SCALAR, []updater{&Kc1, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc1.cpuLUT(), Msat.cpuLUT())
	})
	//kc2_red = Kc2 / Msat
	kc2_red.init(SCALAR, []updater{&Kc2, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc2.cpuLUT(), Msat.cpuLUT())
	})
	//kc3_red = Kc3 / Msat
	kc3_red.init(SCALAR, []updater{&Kc3, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, Kc3.cpuLUT(), Msat.cpuLUT())
	})
}

func addUniaxialAnisotropyField(dst *data.Slice) {
	if ku1_red.nonZero() || ku2_red.nonZero() {
		cuda.AddUniaxialAnisotropy(dst, M.Buffer(), ku1_red.gpuLUT1(), ku2_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
	}
}

func addCubicAnisotropyField(dst *data.Slice) {
	if kc1_red.nonZero() || kc2_red.nonZero() || kc3_red.nonZero() {
		cuda.AddCubicAnisotropy(dst, M.Buffer(), kc1_red.gpuLUT1(), kc2_red.gpuLUT1(), kc3_red.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
	}
}

// Add the anisotropy field to dst
func AddAnisotropyField(dst *data.Slice) {
	addUniaxialAnisotropyField(dst)
	addCubicAnisotropyField(dst)
}

func AddAnisotropyEnergyDensity(dst *data.Slice) {
	haveUnixial := ku1_red.nonZero() || ku2_red.nonZero()
	haveCubic := kc1_red.nonZero() || kc2_red.nonZero() || kc3_red.nonZero()

	if !haveUnixial && !haveCubic {
		return
	}

	buf := cuda.Buffer(B_anis.NComp(), B_anis.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf, r := M_full.Slice()
	if r {
		defer cuda.Recycle(Mf)
	}

	if haveUnixial {
		// 1st
		cuda.Zero(buf)
		cuda.AddUniaxialAnisotropy(buf, M.Buffer(), ku1_red.gpuLUT1(), zero.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -1./2., buf, Mf)

		// 2nd
		cuda.Zero(buf)
		cuda.AddUniaxialAnisotropy(buf, M.Buffer(), zero.gpuLUT1(), ku2_red.gpuLUT1(), AnisU.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -1./4., buf, Mf)
	}

	if haveCubic {
		// 1st
		cuda.Zero(buf)
		cuda.AddCubicAnisotropy(buf, M.Buffer(), kc1_red.gpuLUT1(), zero.gpuLUT1(), zero.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -1./4., buf, Mf)

		// 2nd
		cuda.Zero(buf)
		cuda.AddCubicAnisotropy(buf, M.Buffer(), zero.gpuLUT1(), kc2_red.gpuLUT1(), zero.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -1./6., buf, Mf)

		// 3nd
		cuda.Zero(buf)
		cuda.AddCubicAnisotropy(buf, M.Buffer(), zero.gpuLUT1(), zero.gpuLUT1(), kc3_red.gpuLUT1(), AnisC1.gpuLUT(), AnisC2.gpuLUT(), regions.Gpu())
		cuda.AddDotProduct(dst, -1./8., buf, Mf)
	}
}

// Returns anisotropy energy in joules.
func GetAnisotropyEnergy() float64 {
	buf := cuda.Buffer(1, Edens_anis.Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddAnisotropyEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}
