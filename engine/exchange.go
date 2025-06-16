package engine

// Exchange interaction (Heisenberg + Dzyaloshinskii-Moriya)
// See also cuda/exchange.cu and cuda/dmi.cu

import (
	"math"
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Aex    = NewScalarParam("Aex", "J/m", "Exchange stiffness", &lex2)
	Dind   = NewScalarParam("Dind", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength", &din2)
	Dbulk  = NewScalarParam("Dbulk", "J/m2", "Bulk Dzyaloshinskii-Moriya strength", &dbulk2)
	lex2   exchParam // inter-cell Aex
	din2   exchParam // inter-cell Dind
	dbulk2 exchParam // inter-cell Dbulk

	B_exch     = NewVectorField("B_exch", "T", "Exchange field", AddExchangeField)
	E_exch     = NewScalarValue("E_exch", "J", "Total exchange energy (including the DMI energy)", GetExchangeEnergy)
	Edens_exch = NewScalarField("Edens_exch", "J/m3", "Total exchange energy density (including the DMI energy density)", AddExchangeEnergyDensity)

	// Average exchange coupling with neighbors. Useful to debug inter-region exchange
	ExchCoupling = NewScalarField("ExchCoupling", "arb.", "Average exchange coupling with neighbors", exchangeDecode)
	DindCoupling = NewScalarField("DindCoupling", "arb.", "Average DMI coupling with neighbors", dindDecode)

	OpenBC = false
)

var AddExchangeEnergyDensity = makeEdensAdder(&B_exch, -0.5) // TODO: normal func

func init() {
	registerEnergy(GetExchangeEnergy, AddExchangeEnergyDensity)
	DeclFunc("ext_ScaleExchange", ScaleInterExchange, "Re-scales exchange coupling between two regions.")
	DeclFunc("ext_InterExchange", InterExchange, "Sets exchange coupling between two regions.")
	DeclFunc("ext_ScaleDind", ScaleInterDind, "Re-scales Dind coupling between two regions.")
	DeclFunc("ext_InterDind", InterDind, "Sets Dind coupling between two regions.")
	DeclVar("OpenBC", &OpenBC, "Use open boundary conditions (default=false)")
	lex2.init(Aex)
	din2.init(Dind)
	dbulk2.init(Dbulk)
}

// Adds the current exchange field to dst
func AddExchangeField(dst *data.Slice) {
	inter := !Dind.isZero()
	bulk := !Dbulk.isZero()
	ms := Msat.MSlice()
	defer ms.Recycle()
	switch {
	case !inter && !bulk:
		cuda.AddExchange(dst, M.Buffer(), lex2.Gpu(), ms, regions.Gpu(), M.Mesh())
	case inter && !bulk:
		Refer("mulkers2017")
		cuda.AddDMI(dst, M.Buffer(), lex2.Gpu(), din2.Gpu(), ms, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
	case bulk && !inter:
		cuda.AddDMIBulk(dst, M.Buffer(), lex2.Gpu(), dbulk2.Gpu(), ms, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		// TODO: add ScaleInterDbulk and InterDbulk
	case inter && bulk:
		util.Fatal("Cannot have interfacial-induced DMI and bulk DMI at the same time")
	}
}

// Set dst to the average exchange coupling per cell (average of lex2 with all neighbors).
func exchangeDecode(dst *data.Slice) {
	cuda.ExchangeDecode(dst, lex2.Gpu(), regions.Gpu(), M.Mesh())
}

// Set dst to the average dmi coupling per cell (average of din2 with all neighbors).
func dindDecode(dst *data.Slice) {
	cuda.ExchangeDecode(dst, din2.Gpu(), regions.Gpu(), M.Mesh())
}

// Returns the current exchange energy in Joules.
func GetExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_exch)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchange(region1, region2 int, scale float64) {
	lex2.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchange(region1, region2 int, value float64) {
	lex2.setInter(region1, region2, value)
}

// Scales the DMI interaction between region 1 and 2.
func ScaleInterDind(region1, region2 int, scale float64) {
	din2.setScale(region1, region2, scale)
}

// Sets the DMI interaction between region 1 and 2.
func InterDind(region1, region2 int, value float64) {
	din2.setInter(region1, region2, value)
}

// stores interregion exchange stiffness and DMI
// the interregion exchange/DMI by default is the harmonic mean (scale=1, inter=0)
type exchParam struct {
	parent         *RegionwiseScalar
	lut            [NREGION * (NREGION + 1) / 2]float32 // harmonic mean of regions (i,j)
	scale          [NREGION * (NREGION + 1) / 2]float32 // extra scale factor for lut[SymmIdx(i, j)]
	inter          [NREGION * (NREGION + 1) / 2]float32 // extra term for lut[SymmIdx(i, j)]
	gpu            cuda.SymmLUT                         // gpu copy of lut, lazily transferred when needed
	gpu_ok, cpu_ok bool                                 // gpu cache up-to-date with lut source
}

// to be called after Aex or scaling changed
func (p *exchParam) invalidate() {
	p.cpu_ok = false
	p.gpu_ok = false
}

func (p *exchParam) init(parent *RegionwiseScalar) {
	for i := range p.scale {
		p.scale[i] = 1 // default scaling
		p.inter[i] = 0 // default additional interexchange term
	}
	p.parent = parent
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *exchParam) Gpu() cuda.SymmLUT {
	p.update()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
}

// sets the interregion exchange/DMI using a specified value (scale = 0)
func (p *exchParam) setInter(region1, region2 int, value float64) {
	p.scale[symmidx(region1, region2)] = float32(0.)
	p.inter[symmidx(region1, region2)] = float32(value)
	p.invalidate()
}

// sets the interregion exchange/DMI by rescaling the harmonic mean (inter = 0)
func (p *exchParam) setScale(region1, region2 int, scale float64) {
	p.scale[symmidx(region1, region2)] = float32(scale)
	p.inter[symmidx(region1, region2)] = float32(0.)
	p.invalidate()
}

func (p *exchParam) update() {
	if !p.cpu_ok {
		ex := p.parent.cpuLUT()
		for i := 0; i < NREGION; i++ {
			exi := ex[0][i]
			for j := i; j < NREGION; j++ {
				exj := ex[0][j]
				I := symmidx(i, j)
				p.lut[I] = p.scale[I]*exchAverage(exi, exj) + p.inter[I]
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

func (p *exchParam) upload() {
	// alloc if needed
	if p.gpu == nil {
		p.gpu = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lut)) * cu.SIZEOF_FLOAT32))
	}
	lut := p.lut // Copy, to work around Go 1.6 cgo pointer limitations.
	cuda.MemCpyHtoD(unsafe.Pointer(p.gpu), unsafe.Pointer(&lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	p.gpu_ok = true
}

// Index in symmetric matrix where only one half is stored.
// (!) Code duplicated in exchange.h
func symmidx(i, j int) int {
	if j <= i {
		return i*(i+1)/2 + j
	} else {
		return j*(j+1)/2 + i
	}
}

// Returns the intermediate value of two exchange/dmi strengths.
// If both arguments have the same sign, the average mean is returned. If the arguments differ in sign
// (which is possible in the case of DMI), the geometric mean of the geometric and arithmetic mean is
// used. This average is continuous everywhere, monotonic increasing, and bounded by the argument values.
func exchAverage(exi, exj float32) float32 {
	if exi*exj >= 0.0 {
		return 2 / (1/exi + 1/exj)
	} else {
		exi_, exj_ := float64(exi), float64(exj)
		sign := math.Copysign(1, exi_+exj_)
		magn := math.Sqrt(math.Sqrt(-exi_*exj_) * math.Abs(exi_+exj_) / 2)
		return float32(sign * magn)
	}
}
