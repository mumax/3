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
	EnableDemag = true                 // enable/disable demag field
	demagconv_  *cuda.DemagConvolution // does the heavy lifting and provides FFTM
)

func init() {
	Msat.init("Msat", "A/m", "Saturation magnetization", []derived{&Bsat, &lex2, &ku1_red, &kc1_red, &temp_red})
	M_full.init(VECTOR, &globalmesh, "m_full", "A/m", "Unnormalized magnetization", SetMFull)
	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	B_demag.init(VECTOR, &globalmesh, "B_demag", "T", "Magnetostatic field", SetDemagField)
	E_demag = NewGetScalar("E_demag", "J", "Magnetostatic energy", getDemagEnergy)
	registerEnergy(getDemagEnergy)

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
		demagConv().Exec(dst, M.Buffer(), vol(), Bsat.gpuLUT1(), regions.Gpu())
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
	if demagconv_ == nil {
		SetBusy("calculating demag kernel")
		defer SetBusy("")
		demagconv_ = cuda.NewDemag(Mesh())
	}
	return demagconv_
}

// Returns the current demag energy in Joules.
func getDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}
