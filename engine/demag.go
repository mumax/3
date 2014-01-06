package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// demag variables
var (
	Msat        ScalarParam
	Bsat        derivedParam
	M_full      setter
	B_demag     setter
	E_demag     *GetScalar
	Edens_demag adder
	EnableDemag = true                 // enable/disable demag field
	conv_       *cuda.DemagConvolution // does the heavy lifting and provides FFTM
)

func init() {
	Msat.init("Msat", "A/m", "Saturation magnetization", []derived{&Bsat, &lex2, &ku1_red, &kc1_red, &temp_red})
	M_full.init(VECTOR, "m_full", "A/m", "Unnormalized magnetization", SetMFull)
	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	B_demag.init(VECTOR, "B_demag", "T", "Magnetostatic field", SetDemagField)
	E_demag = NewGetScalar("E_demag", "J", "Magnetostatic energy", GetDemagEnergy)
	Edens_demag.init(SCALAR, "Edens_demag", "J/m3", "Exchange energy density (normal+DM)", addEdens(&B_demag, -0.5))
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
	msat, r := Msat.Slice()
	if r {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M.Buffer().Comp(c), msat)
	}
}

// returns demag convolution, making sure it's initialized
func demagConv() *cuda.DemagConvolution {
	if conv_ == nil {
		LogOutput("calculating demag kernel")
		defer LogOutput("kernel done")
		GUI.SetBusy(true)
		defer GUI.SetBusy(false)
		conv_ = cuda.NewDemag(Mesh())
	}
	return conv_
}

// Returns the current demag energy in Joules.
func GetDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}
