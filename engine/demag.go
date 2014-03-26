package engine

// Calculation of magnetostatic field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// Demag variables
var (
	Msat          ScalarParam
	Bsat          derivedParam
	M_full        vSetter
	B_demag       vSetter
	E_demag       *GetScalar
	Edens_demag   sAdder
	EnableDemag   = true                 // enable/disable demag field
	conv_         *cuda.DemagConvolution // does the heavy lifting and provides FFTM
	DemagAccuracy = 6.0                  // Demag accuracy (divide cubes in at most N^3 points)
	CacheDir      = ""                   // directory for kernel cache
)

func init() {
	Msat.init("Msat", "A/m", "Saturation magnetization", []derived{&Bsat, &lex2, &ku1_red, &kc1_red, &temp_red})
	M_full.init("m_full", "A/m", "Unnormalized magnetization", SetMFull)
	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	DeclVar("DemagAccuracy", &DemagAccuracy, "Controls accuracy of demag kernel")
	B_demag.init("B_demag", "T", "Magnetostatic field", SetDemagField)
	E_demag = NewGetScalar("E_demag", "J", "Magnetostatic energy", GetDemagEnergy)
	Edens_demag.init("Edens_demag", "J/m3", "Exchange energy density (normal+DM)", addEdens(&B_demag, -0.5))
	registerEnergy(GetDemagEnergy, Edens_demag.AddTo)

	//Bsat = Msat * mu0
	Bsat.init(SCALAR, []updater{&Msat}, func(p *derivedParam) {
		Ms := Msat.cpuLUT()
		for i, ms := range Ms[0] {
			p.cpu_buf[0][i] = mag.Mu0 * ms
		}
	})
}

// Sets dst to the current demag field
func SetDemagField(dst *data.Slice) {
	if EnableDemag {
		demagConv().Exec(dst, M.Buffer(), geometry.Gpu(), Bsat.gpuLUT1(), regions.Gpu())
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

// Sets dst to the full (unnormalized) magnetization in A/m
func SetMFull(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

// returns demag convolution, making sure it's initialized
func demagConv() *cuda.DemagConvolution {
	if conv_ == nil {
		SetBusy(true)
		defer SetBusy(false)
		kernel := mag.DemagKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, CacheDir)
		conv_ = cuda.NewDemag(Mesh().Size(), Mesh().PBC(), kernel)
	}
	return conv_
}

// Returns the current demag energy in Joules.
func GetDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}
