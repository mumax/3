package engine

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

var (
	Aex        ScalarParam // Exchange stiffness
	Dex        ScalarParam // DMI strength
	B_exch     vAdder      // exchange field (T) output handle
	lex2       exchParam   // inter-cell exchange in 2e18 * Aex / Msat
	E_exch     *GetScalar
	Edens_exch sAdder
)

func init() {
	Aex.init("Aex", "J/m", "Exchange stiffness", []derived{&lex2})
	Dex.init("Dex", "J/m2", "Dzyaloshinskii-Moriya strength", []derived{})
	B_exch.init("B_exch", "T", "Exchange field", AddExchangeField)
	E_exch = NewGetScalar("E_exch", "J", "Exchange energy (normal+DM)", GetExchangeEnergy)
	Edens_exch.init("Edens_exch", "J/m3", "Exchange energy density (normal+DM)", addEdens(&B_exch, -0.5))
	registerEnergy(GetExchangeEnergy, Edens_exch.AddTo)
	DeclFunc("ScaleExchReg", ScaleInterExchange, "Re-scales exchange coupling between two regions.")
	lex2.init()
}

// Adds the current exchange field to dst
func AddExchangeField(dst *data.Slice) {
	if Dex.isZero() {
		cuda.AddExchange(dst, M.Buffer(), lex2.Gpu(), regions.Gpu(), M.Mesh())
	} else {
		// DMI only implemented for uniform parameters
		// interaction not clear with space-dependent parameters
		util.AssertMsg(Msat.IsUniform() && Aex.IsUniform() && Dex.IsUniform(),
			"DMI: Msat, Aex, Dex must be uniform")
		msat := Msat.GetRegion(0)
		D := Dex.GetRegion(0)
		A := Aex.GetRegion(0) / msat
		cuda.AddDMI(dst, M.Buffer(), float32(D/msat), float32(D/msat), 0, float32(A), M.Mesh()) // dmi+exchange
	}
}

// Returns the current exchange energy in Joules.
// Note: the energy is defined up to an arbitrary constant,
// ground state energy is not necessarily zero or comparable
// to other simulation programs.
func GetExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_exch)
}

// Defines the exchange coupling between different regions by specifying the
// exchange length of the interaction between them.
// 	lex := sqrt(2*Aex / Msat)
// In case of different materials it is not always clear what the exchange
// between them should be, especially if they have different Msat. By specifying
// the exchange length, it is up to the user to decide which Msat to use.
// A negative length may be specified to obtain antiferromagnetic coupling.
func ScaleInterExchange(region1, region2 int, scale float64) {
	lex2.scale[symmidx(region1, region2)] = float32(scale)
	lex2.invalidate()
}

// stores interregion exchange stiffness
type exchParam struct {
	lut, scale     [NREGION * (NREGION + 1) / 2]float32 // cpu lookup-table
	gpu            cuda.SymmLUT                         // gpu copy of lut, lazily transferred when needed
	gpu_ok, cpu_ok bool                                 // gpu cache up-to date with lut source
}

func (p *exchParam) invalidate() {
	p.cpu_ok = false
	p.gpu_ok = false
}

func (p *exchParam) init() {
	for i := range p.scale {
		p.scale[i] = 1
	}
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

func (p *exchParam) update() {
	if !p.cpu_ok {
		msat := Msat.cpuLUT()
		aex := Aex.cpuLUT()

		// todo: conditional
		for i := 0; i < NREGION; i++ {
			lexi := 2e18 * safediv(aex[0][i], msat[0][i])
			for j := 0; j <= i; j++ {
				lexj := 2e18 * safediv(aex[0][j], msat[0][j])
				I := symmidx(i, j)
				p.lut[I] = p.scale[I] * 2 / (1/lexi + 1/lexj)
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
	cuda.Sync()
	cu.MemcpyHtoD(cu.DevicePtr(p.gpu), unsafe.Pointer(&p.lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	cuda.Sync()
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

//func (p *exchParam) String() string {
//	str := ""
//	for j := 0; j < NREGION; j++ {
//		for i := 0; i <= j; i++ {
//			str += fmt.Sprint(p.lut[symmidx(i, j)], "\t")
//		}
//		str += "\n"
//	}
//	return str
//}
