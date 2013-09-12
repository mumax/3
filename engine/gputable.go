package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

// TODO: fuse cputable

type gpuTable struct {
	gpu_buf cuda.LUTPtrs // gpu copy of lut, lazily transferred when needed
	gpu_ok  bool         // gpu cache up-to date with lut source
	source  interface {
		Cpu() [][NREGION]float32
	}
}

func (p *gpuTable) init(nComp int) {
	p.gpu_buf = make(cuda.LUTPtrs, nComp)
}

func (p *gpuTable) LUT() cuda.LUTPtrs {
	if !p.gpu_ok {
		p.upload(p.source.Cpu())
	}
	return p.gpu_buf
}

func (p *gpuTable) LUT1() cuda.LUTPtr {
	util.Assert(len(p.gpu_buf) == 1)
	return cuda.LUTPtr(p.LUT()[0])
}

func (p *gpuTable) Get() (*data.Slice, bool) {
	gpu := p.LUT()
	nComp := len(p.gpu_buf)
	b := cuda.GetBuffer(nComp, &globalmesh)
	for c := 0; c < nComp; c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
	}
	return b, true
}

func (p *gpuTable) upload(cpu [][NREGION]float32) {
	util.Assert(len(cpu) == len(p.gpu_buf))
	ncomp := len(cpu)
	p.assureAlloc()
	for c2 := range p.gpu_buf {
		c := util.SwapIndex(c2, ncomp) // XYZ swap here
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu_buf[c]), unsafe.Pointer(&cpu[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
	}
	p.gpu_ok = true
}

func (p *gpuTable) assureAlloc() {
	if p.gpu_buf[0] == nil {
		for i := range p.gpu_buf {
			p.gpu_buf[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}
}

type cpuTable struct {
	cpu_buf [][NREGION]float32 // look-up table source
}

func (b *cpuTable) NComp() int     { return len(b.cpu_buf) }
func (b *cpuTable) init(ncomp int) { b.cpu_buf = make([][NREGION]float32, ncomp) }
