package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

type lut struct {
	gpu_buf cuda.LUTPtrs       // gpu copy of lut, lazily transferred when needed
	gpu_ok  bool               // gpu cache up-to date with lut source
	cpu_buf [][NREGION]float32 // look-up table source
	source  updater
}

type updater interface {
	update()
}

func (p *lut) init(nComp int, source updater) {
	p.gpu_buf = make(cuda.LUTPtrs, nComp)
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.source = source
}

func (p *lut) LUT() cuda.LUTPtrs {
	p.source.update()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu_buf
}

func (p *lut) LUT1() cuda.LUTPtr {
	util.Assert(len(p.gpu_buf) == 1)
	return cuda.LUTPtr(p.LUT()[0])
}

func (p *lut) Get() (*data.Slice, bool) {
	gpu := p.LUT()
	nComp := len(p.gpu_buf)
	b := cuda.GetBuffer(nComp, &globalmesh)
	for c := 0; c < nComp; c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
	}
	return b, true
}

func (p *lut) upload() {
	cpu := p.cpu_buf
	util.Assert(len(cpu) == len(p.gpu_buf))
	ncomp := len(cpu)
	p.assureAlloc()
	for c2 := range p.gpu_buf {
		c := util.SwapIndex(c2, ncomp) // XYZ swap here
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu_buf[c]), unsafe.Pointer(&cpu[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
	}
	p.gpu_ok = true
	log.Println("upload", cpu)
}

func (p *lut) assureAlloc() {
	if p.gpu_buf[0] == nil {
		for i := range p.gpu_buf {
			p.gpu_buf[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}
}

func (b *lut) NComp() int { return len(b.cpu_buf) }
