package engine

// Exchange interaction (Heisenberg + Dzyaloshinskii-Moriya)
// See also cuda/exchange.cu and cuda/dmi.cu

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

var (
	Aex          ScalarParam // Exchange stiffness
	Dind         ScalarParam // interfacial DMI strength
	Dbulk        ScalarParam // bulk DMI strength
	B_exch       vAdder      // exchange field (T) output handle
	lex2         aexchParam  // inter-cell exchange in 1e18 * Aex / Msat
	din2         dexchParam  // inter-cell interfacial DMI in 1e9 * Dex / Msat
	dbulk2       dexchParam  // inter-cell bulk DMI in 1e9 * Dex / Msat
	E_exch       *GetScalar  // Exchange energy
	Edens_exch   sAdder      // Exchange energy density
	ExchCoupling sSetter     // Average exchange coupling with neighbors. Useful to debug inter-region exchange
)

func init() {
	Aex.init("Aex", "J/m", "Exchange stiffness", []derived{&lex2})
	Dind.init("Dind", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength", []derived{&din2})
	Dbulk.init("Dbulk", "J/m2", "Bulk Dzyaloshinskii-Moriya strength", []derived{&dbulk2})
	B_exch.init("B_exch", "T", "Exchange field", AddExchangeField)
	E_exch = NewGetScalar("E_exch", "J", "Exchange energy (normal+DM)", GetExchangeEnergy)
	Edens_exch.init("Edens_exch", "J/m3", "Exchange energy density (normal+DM)", makeEdensAdder(&B_exch, -0.5))
	registerEnergy(GetExchangeEnergy, Edens_exch.AddTo)
	DeclFunc("ext_ScaleExchange", ScaleInterExchange, "Re-scales exchange coupling between two regions.")
	lex2.init()
	din2.init(&Dind)
	dbulk2.init(&Dbulk)
	ExchCoupling.init("ExchCoupling", "arb.", "Average exchange coupling with neighbors", exchangeDecode)
}

// Adds the current exchange field to dst
func AddExchangeField(dst *data.Slice) {
	inter := !Dind.isZero()
	bulk := !Dbulk.isZero()
	switch {
	case !inter && !bulk:
		cuda.AddExchange(dst, M.Buffer(), lex2.Gpu(), regions.Gpu(), M.Mesh())
	case inter && !bulk:
		// DMI kernel has space-dependent parameters, but
		// correct averaging between regions not yet clear nor tested, so disallow.
		util.AssertMsg(allowUnsafe || (Msat.IsUniform() && Aex.IsUniform() && Dind.IsUniform()), "DMI: Msat, Aex, Dex must be uniform")
		cuda.AddDMI(dst, M.Buffer(), lex2.Gpu(), din2.Gpu(), regions.Gpu(), M.Mesh()) // dmi+exchange
	case bulk && !inter:
		util.AssertMsg(allowUnsafe || (Msat.IsUniform() && Aex.IsUniform() && Dbulk.IsUniform()), "DMI: Msat, Aex, Dex must be uniform")
		cuda.AddDMIBulk(dst, M.Buffer(), lex2.Gpu(), dbulk2.Gpu(), regions.Gpu(), M.Mesh()) // dmi+exchange
	case inter && bulk:
		util.Fatal("Cannot have induced and interfacial DMI at the same time")
	}
}

// Set dst to the average exchange coupling per cell (average of lex2 with all neighbors).
func exchangeDecode(dst *data.Slice) {
	cuda.ExchangeDecode(dst, lex2.Gpu(), regions.Gpu(), M.Mesh())
}

// Returns the current exchange energy in Joules.
func GetExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_exch)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex/Msat.
func ScaleInterExchange(region1, region2 int, scale float64) {
	lex2.scale[symmidx(region1, region2)] = float32(scale)
	lex2.invalidate()
}

// stores interregion exchange stiffness
type exchParam struct {
	lut            [NREGION * (NREGION + 1) / 2]float32 // 1e18 * harmonic mean of Aex/Msat in regions (i,j)
	scale          [NREGION * (NREGION + 1) / 2]float32 // extra scale factor for lut[SymmIdx(i, j)]
	gpu            cuda.SymmLUT                         // gpu copy of lut, lazily transferred when needed
	gpu_ok, cpu_ok bool                                 // gpu cache up-to date with lut source
}

// to be called after Aex, Msat or scaling changed
func (p *exchParam) invalidate() {
	p.cpu_ok = false
	p.gpu_ok = false
}

func (p *aexchParam) init() {
	for i := range p.scale {
		p.scale[i] = 1 // default scaling
	}
}

func (p *dexchParam) init(parent *ScalarParam) {
	for i := range p.scale {
		p.scale[i] = 1 // default scaling
	}
	p.parent = parent
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *dexchParam) Gpu() cuda.SymmLUT {
	p.update()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
}

func (p *aexchParam) Gpu() cuda.SymmLUT {
	p.update()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
	// TODO: dedup
}

type aexchParam struct{ exchParam }
type dexchParam struct {
	parent *ScalarParam
	exchParam
}

func (p *aexchParam) update() {
	if !p.cpu_ok {
		msat := Msat.cpuLUT()
		aex := Aex.cpuLUT()

		for i := 0; i < NREGION; i++ {
			lexi := 1e18 * safediv(aex[0][i], msat[0][i])
			for j := i; j < NREGION; j++ {
				lexj := 1e18 * safediv(aex[0][j], msat[0][j])
				I := symmidx(i, j)
				p.lut[I] = p.scale[I] * 2 / (1/lexi + 1/lexj)
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

func (p *dexchParam) update() {
	if !p.cpu_ok {
		msat := Msat.cpuLUT()
		dex := p.parent.cpuLUT()

		for i := 0; i < NREGION; i++ {
			dexi := 1e9 * safediv(dex[0][i], msat[0][i])
			for j := i; j < NREGION; j++ {
				dexj := 1e9 * safediv(dex[0][j], msat[0][j])
				I := symmidx(i, j)
				p.lut[I] = p.scale[I] * 2 / (1/dexi + 1/dexj)
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

func (p *exchParam) upload() {
	// alloc if  needed
	if p.gpu == nil {
		p.gpu = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lut)) * cu.SIZEOF_FLOAT32))
	}
	cuda.MemCpyHtoD(unsafe.Pointer(p.gpu), unsafe.Pointer(&p.lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	p.gpu_ok = true
}

// Index in symmetric matrix where only one half is stored.
// (!) Code duplicated in exchange.cu
func symmidx(i, j int) int {
	if j <= i {
		return i*(i+1)/2 + j
	} else {
		return j*(j+1)/2 + i
	}
}
