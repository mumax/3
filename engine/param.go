package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

// param stores a space-dependent material parameter,
// by keeping a look-up table mapping region index to float value.
type param struct {
	gpu_ok  bool               // gpu cache up-to date with lut source
	gpu_buf cuda.LUTPtrs       // gpu copy of lut, lazily transferred when needed
	cpu_buf [][NREGION]float32 // look-up table source
	update  func()             // updates cpu_buf, if needed
	doc
}

func (p *param) init_(nComp int, name, unit string, upd func()) {
	util.Assert(nComp > 0)
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.update = upd
	p.doc = doc{nComp: nComp, name: name, unit: unit}
}

func (p *param) Cpu() [][NREGION]float32 {
	p.update()
	return p.cpu_buf
}

// Get a GPU mirror of the look-up table, copies to GPU if needed.
func (p *param) gpu() cuda.LUTPtrs {
	if !p.gpu_ok {
		p.upload()
	}
	log.Println(p.name, ".gpu:", p.gpu_buf)
	return p.gpu_buf
}

func (p *param) Gpu1() cuda.LUTPtr {
	return cuda.LUTPtr(p.gpu()[0])
}
func (p *param) Gpu3() cuda.LUTPtrs { return p.gpu() }

// upload cpu table to gpu
func (p *param) upload() {
	p.assureAlloc()
	_ = p.Cpu() // assures cpu is up-to-date, but don't leak pointer to heap
	log.Println("upload", p.Name())
	for c2 := range p.gpu_buf {
		c := util.SwapIndex(c2, p.NComp()) // XYZ swap here
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu_buf[c]), unsafe.Pointer(&p.cpu_buf[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
	}
	p.gpu_ok = true
}

// allocte if when needed
func (p *param) assureAlloc() {
	if p.gpu == nil {
		log.Println("alloc", p.name)
		util.Assert(p.NComp() > 0)
		p.gpu_buf = make(cuda.LUTPtrs, p.NComp())
		for i := range p.gpu_buf {
			p.gpu_buf[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
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

func (p *param) Mesh() *data.Mesh {
	return &globalmesh
}
