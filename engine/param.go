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
type param struct {
	gpu_ok  bool               // gpu cache up-to date with lut source
	gpu_buf cuda.LUTPtrs       // gpu copy of lut, lazily transferred when needed
	cpu_buf [][NREGION]float32 // look-up table source
	update  func()             // updates cpu_buf, if needed
	modtime float64            // timestamp for cpu data
	doc
}

func (p *param) init_(nComp int, name, unit string, upd func()) {
	util.Assert(nComp > 0)
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.update = upd
	p.doc = doc{nComp: nComp, name: name, unit: unit}
	p.modtime = math.Inf(-1)
}

func (p *param) Cpu() (data [][NREGION]float32, modtime float64) {
	p.update()
	return p.cpu_buf, p.modtime
}
func (p *param) Cpu1() (data [NREGION]float32, modtime float64) {
	util.Assert(p.nComp == 1)
	comps, t := p.Cpu()
	return comps[0], t
}

// Get a GPU mirror of the look-up table, copies to GPU if needed.
func (p *param) gpu() cuda.LUTPtrs {
	if !p.gpu_ok {
		p.upload()
	}
	log.Println("gpu:", p.name, ".gpu:", p.gpu_buf)
	return p.gpu_buf
}

func (p *param) Gpu1() cuda.LUTPtr {
	log.Println("Gpu1:")
	ptr := cuda.LUTPtr(p.gpu()[0])
	log.Println("Gpu1:", ptr)
	return ptr
}

func (p *param) Gpu3() cuda.LUTPtrs { return p.gpu() }

// upload cpu table to gpu
func (p *param) upload() {
	p.assureAlloc()
	_, _ = p.Cpu() // assures cpu is up-to-date, but don't leak pointer to heap
	log.Println("upload", p.Name())
	for c2 := range p.gpu_buf {
		c := util.SwapIndex(c2, p.NComp()) // XYZ swap here
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu_buf[c]), unsafe.Pointer(&p.cpu_buf[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
	}
	p.gpu_ok = true
	log.Println("upload", p.name, "gpu_buf=", p.gpu_buf)
}

// allocte if when needed
func (p *param) assureAlloc() {
	util.Assert(p.NComp() > 0)
	if p.gpu_buf == nil {
		log.Println("alloc", p.name)
		p.gpu_buf = make(cuda.LUTPtrs, p.NComp())
		for i := range p.gpu_buf {
			p.gpu_buf[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}
}

func (p *param) zero() bool {
	cpu, _ := p.Cpu()
	for c := range cpu {
		for r := 0; r < regions.maxreg; r++ {
			if cpu[c][r] != 0 {
				return false
			}
		}
	}
	return true
}

func (p *param) Mesh() *data.Mesh {
	return &globalmesh
}
