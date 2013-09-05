package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"math"
	"unsafe"
)

// param stores a space-dependent material parameter,
// by keeping a look-up table mapping region index to float value.
// implementation allows arbitrary number of components,
// narrowed down to 1 or 3 in ScalarParam and VectorParam
type param struct {
	gpu_ok    bool               // gpu cache up-to date with lut source
	gpu       cuda.LUTPtrs       // gpu copy of lut, lazily transferred when needed
	cpu_stamp float64            // timestamp for cpu data
	cpu       [][NREGION]float32 // look-up table source
	upd       [NREGION]func()    // per-region time-dependent functions
	updateAll func()             // master updater, may be nil
	deps      []*param           // dependencies
	doc
}

// constructor: TODO: init + decl
func newParam(nComp int, name, unit string, deps ...*param) param {
	return param{
		gpu:       make(cuda.LUTPtrs, nComp),
		cpu_stamp: math.Inf(-1),
		cpu:       make([][NREGION]float32, nComp),
		deps:      deps,
		doc:       doc{nComp: nComp, name: name, unit: unit}}
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *param) Gpu() cuda.LUTPtrs {
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
}

// XYZ swap here
func (p *param) upload() {
	// alloc when needed, allows param use in init()
	if p.gpu[0] == nil {
		for i := range p.gpu {
			p.gpu[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}

	// assures cpu is up-to-date
	cpu := p.Cpu()

	// upload
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp())
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&cpu[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
	}
	p.gpu_ok = true
}

func (p *param) Cpu() [][NREGION]float32 {
	if !p.cpu_upToDate() {
		p.updateCPU()
	}
	return p.cpu
}

func (p *param) cpu_upToDate() bool {
	if p.cpu_stamp != Time {
		return false
	}
	for _, d := range p.deps {
		if !d.cpu_upToDate() {
			return false
		}
	}
	return true
}

func (p *param) updateCPU() {
	p.cpu_stamp = Time
	p.gpu_ok = false

	if p.updateAll != nil {
		p.updateAll()
	} else {
		for r := 0; r < regions.maxreg; r++ {
			if p.upd[r] != nil {
				p.upd[r]()
			}
		}
	}
}

func (p *param) zero() bool {
	cpu := p.Cpu()
	for c := range cpu {
		for r := 0; r < regions.maxreg; r++ {
			if cpu[c][r] != 0 {
				return false
			}
		}
	}
	return true
}

func (p *param) setRegion(region int, v ...float64) {
	util.Argument(len(v) == p.NComp()) // note: also panics if param not initialized (nComp = 0)

	p.gpu_ok = false
	p.upd[region] = nil
	for c := range v {
		p.cpu[c][region] = float32(v[c])
	}
}

// set in all regions except 0
func (p *param) setUniform(v ...float64) {
	for r := 1; r < NREGION; r++ {
		p.setRegion(r, v...)
	}
}

func (p *param) GetRegion(region int) []float64 {
	cpu := p.Cpu()
	v := make([]float64, p.nComp)
	for i := range v {
		v[i] = float64(cpu[i][region])
	}
	return v
}

func (p *param) GetVec() []float64 {
	return p.GetRegion(1)
}

func (p *param) getUniform() []float64 {
	cpu := p.Cpu()
	v := make([]float64, p.NComp())
	for c := range v {
		x := cpu[c][1]
		for r := 2; r < regions.maxreg; r++ {
			if p.cpu[c][r] != x {
				log.Panicf("%v is not uniform", p.name)
			}
		}
		v[c] = float64(x)
	}
	return v
}

func (p *param) Get() (*data.Slice, bool) {
	p.upload()
	b := cuda.GetBuffer(p.NComp(), p.Mesh())
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(p.gpu[c]), regions.Gpu())
	}
	return b, true
}

func (p *param) Mesh() *data.Mesh { return &globalmesh }
