package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

// look-up table for region based parameters
type lut struct {
	gpu_buf cuda.LUTPtrs       // gpu copy of cpu buffer, only transferred when needed
	gpu_ok  bool               // gpu cache up-to date with cpu source?
	cpu_buf [][NREGION]float32 // table data on cpu
	source  updater            // updates cpu data
	nUpload int                // debug counters for number of uploads TODO: rm
}

type updater interface {
	update() // updates cpu lookup table
}

func (p *lut) init(nComp int, source updater) {
	p.gpu_buf = make(cuda.LUTPtrs, nComp)
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.source = source
}

// get an up-to-date version of the lookup-table on CPU
func (p *lut) CpuLUT() [][NREGION]float32 {
	p.source.update()
	return p.cpu_buf
}

// get an up-to-date version of the lookup-table on GPU
func (p *lut) LUT() cuda.LUTPtrs {
	p.source.update()
	if !p.gpu_ok {
		// upload to GPU
		p.assureAlloc()
		ncomp := p.NComp()
		for c2 := range p.gpu_buf {
			c := util.SwapIndex(c2, ncomp) // XYZ swap here
			cu.MemcpyHtoD(cu.DevicePtr(p.gpu_buf[c]), unsafe.Pointer(&p.cpu_buf[c2][0]), cu.SIZEOF_FLOAT32*NREGION)
		}
		p.gpu_ok = true
		p.nUpload++
	}
	return p.gpu_buf
}

// utility for LUT of single-component data
func (p *lut) LUT1() cuda.LUTPtr {
	util.Assert(len(p.gpu_buf) == 1)
	return cuda.LUTPtr(p.LUT()[0])
}

// all data is 0?
func (p *lut) isZero() bool {
	v := p.CpuLUT()
	for c := range v {
		for i := range v[c] { // TODO: regions.maxreg
			if v[c][i] != 0 {
				return false
			}
		}
	}
	return true
}

func (p *lut) assureAlloc() {
	if p.gpu_buf[0] == nil {
		for i := range p.gpu_buf {
			p.gpu_buf[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}
}

func (b *lut) NComp() int   { return len(b.cpu_buf) }
func (b *lut) NUpload() int { return b.nUpload }

// uncompress the table to a full array with parameter values per cell.
func (p *lut) Get() (*data.Slice, bool) {
	gpu := p.LUT()
	b := cuda.Buffer(p.NComp(), &globalmesh)
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
	}
	return b, true
}
