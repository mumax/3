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
	doc
}

func (p *param) init_(nComp int, name, unit string) {
	return param{
		cpu_stamp: math.Inf(-1),
		cpu:       make([][NREGION]float32, nComp),
		deps:      deps,
		doc:       doc{nComp: nComp, name: name, unit: unit}}
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *param) gpu() cuda.LUTPtrs {
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
}

func (p *param) Gpu1() cuda.LUTPtr {
	return cuda.LUTPtr(p.gpu()[0])
}

func (p *param) Gpu3() cuda.LUTPtr {
	return p.gpu()
}

// upload cpu table to gpu
func (p *param) upload() {
	p.assureAlloc()
	cpu := p.Cpu() // assures cpu is up-to-date
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp()) // XYZ swap here
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&cpu[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
	}
	p.gpu_ok = true
}

// allocte if when needed
func (p *param) assureAlloc() {
	if p.gpu == nil {
		util.Assert(p.NComp() > 0)
		p.gpu = make(cuda.LUTPtrs, p.NComp())
		for i := range p.gpu {
			p.gpu[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}
}

type inputParam struct {
	param
	update [NREGION]func() []float64
}

func (p *inputParam) Cpu() [][NREGION]float32 {
	if p.cpu_stamp != Time {
		p.cpu_stamp = Time
		p.gpu_ok = false

		for r := 0; r < regions.maxreg; r++ {
			if p.upd[r] != nil {
				v := upd[r]()
				for c := range p.cpu {
					p.cpu[c][r] = v[c]
				}
			}
		}
	}
	return p.cpu
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

func (p *inputParam) setRegion(region int, v ...float64) {
	util.Argument(len(v) == p.NComp()) // note: also panics if param not initialized (nComp = 0)

	p.gpu_ok = false
	p.upd[region] = nil
	for c := range v {
		p.cpu[c][region] = float32(v[c])
	}
}

// set in all regions except 0
func (p *inputParam) setUniform(v ...float64) {
	for r := 1; r < NREGION; r++ {
		p.setRegion(r, v...)
	}
}

func (p *inputParam) GetRegion(region int) []float64 {
	cpu := p.Cpu()
	v := make([]float64, p.nComp)
	for i := range v {
		v[i] = float64(cpu[i][region])
	}
	return v
}

func (p *inputParam) GetVec() []float64 {
	return p.GetRegion(1)
}

func (p *inputParam) getUniform() []float64 {
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

func (p *inputParam) Get() (*data.Slice, bool) {
	p.upload()
	b := cuda.GetBuffer(p.NComp(), p.Mesh())
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(p.gpu[c]), regions.Gpu())
	}
	return b, true
}

func (p *inputParam) Mesh() *data.Mesh { return &globalmesh }
