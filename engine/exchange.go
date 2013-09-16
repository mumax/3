package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

var (
	Aex, Dex ScalarParam
	B_exch   adder     // exchange field (T) output handle
	lex2     exchParam // inter-cell exchange length squared * 1e18
	E_exch   = NewGetScalar("E_exch", "J", "Exchange energy (normal+DM)", getExchangeEnergy)
)

func init() {
	Aex.init("Aex", "J/m", "Exchange stiffness", []derived{&lex2})
	Dex.init("Dex", "J/m2", "Dzyaloshinskii-Moriya strength", nil)
	DeclFunc("setLexchange", SetLExchange, "Sets inter-material exchange length between two regions.")

	B_exch.init(3, &globalmesh, "B_exch", "T", "Exchange field (T)", func(dst *data.Slice) {
		if Dex.isZero() {
			cuda.AddExchange(dst, M.buffer, lex2.Gpu(), regions.Gpu())
		} else {
			// DMI only implemented for uniform parameters
			// interaction not clear with space-dependent parameters
			msat := Msat.getUniform()[0]
			D := Dex.getUniform()[0] / msat
			A := Aex.getUniform()[0] / msat
			cuda.AddDMI(dst, M.buffer, float32(D), float32(A)) // dmi+exchange
		}
	})

	registerEnergy(getExchangeEnergy)
}

// Returns the current exchange energy in Joules.
// Note: the energy is defined up to an arbitrary constant,
// ground state energy is not necessarily zero or comparable
// to other simulation programs.
func getExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_exch)
}

// Defines the exchange coupling between different regions by specifying the
// exchange length of the interaction between them.
// 	lex := sqrt(2*Aex / Msat)
// In case of different materials it is not always clear what the exchange
// between them should be, especially if they have different Msat. By specifying
// the exchange length, it is up to the user to decide which Msat to use.
// When using regions, there is by default no exchange coupling between different regions.
// A negative length may be specified to obtain antiferromagnetic coupling.
func SetLExchange(region1, region2 int, exlen float64) {
	l2 := sign(exlen) * (exlen * exlen) * 1e18
	lex2.SetInterRegion(region1, region2, l2)
}

func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

type exchParam struct {
	lut            [NREGION * (NREGION + 1) / 2]float32 // look-up table source
	gpu            cuda.SymmLUT                         // gpu copy of lut, lazily transferred when needed
	gpu_ok, cpu_ok bool                                 // gpu cache up-to date with lut source

}

func (p *exchParam) invalidate() { p.cpu_ok = false }

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
		msat := Msat.CpuLUT()
		aex := Aex.CpuLUT()

		// todo: conditional
		for i := 0; i < regions.maxreg; i++ {
			lexi := 2e18 * safediv(aex[0][i], msat[0][i])
			for j := 0; j <= i; j++ {
				lexj := 2e18 * safediv(aex[0][j], msat[0][j])
				p.lut[symmidx(i, j)] = 2 / (1/lexi + 1/lexj)
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

func safediv(a, b float32) float32 {
	if b == 0 {
		return 0
	} else {
		return a / b
	}
}

// XYZ swap here
func (p *exchParam) upload() {
	// alloc if  needed
	if p.gpu == nil {
		p.gpu = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lut)) * cu.SIZEOF_FLOAT32))
	}
	log.Println("upload Aex")
	cu.MemcpyHtoD(cu.DevicePtr(p.gpu), unsafe.Pointer(&p.lut[0]), cu.SIZEOF_FLOAT32*int64(len(p.lut)))
	p.gpu_ok = true
}

func (p *exchParam) SetInterRegion(r1, r2 int, val float64) {
	v := float32(val)
	p.lut[symmidx(r1, r2)] = v

	if r1 == r2 {
		r := r1
		for i := 0; i < NREGION; i++ {
			if p.lut[symmidx(i, i)] == v {
				p.lut[symmidx(r, i)] = v
			} else {
				p.lut[symmidx(r, i)] = 0
			}
		}
	}

	p.gpu_ok = false
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

//func (p *exchParam) SetUniform(v float64) {
//	for i := range p.lut {
//		p.lut[i] = float32(v)
//	}
//	p.ok = false
//}

//func (p *exchParam) getInter(r1, r2 int) float64 {
//	return float64(p.lut[symmidx(r1, r2)])
//}
//
func (p *exchParam) String() string {
	str := ""
	for j := 0; j < regions.maxreg; j++ {
		for i := 0; i <= j; i++ {
			str += fmt.Sprint(p.lut[symmidx(i, j)], "\t")
		}
		str += "\n"
	}
	return str
}
