package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"log"
	"unsafe"
)

const LUTSIZE = 256

type Param struct {
	gpu         cuda.LUTPtrs
	cpu         [][LUTSIZE]float32
	ok          bool
	post_update func(region int) // called after region value changed
	autosave
}

func param(nComp int, name, unit string) Param {
	var p Param
	p.autosave = newAutosave(nComp, name, unit, Mesh())
	p.gpu = make(cuda.LUTPtrs, nComp)
	for i := range p.gpu {
		p.gpu[i] = cuda.MemAlloc(LUTSIZE * cu.SIZEOF_FLOAT32)
	}
	p.cpu = make([][LUTSIZE]float32, nComp)
	p.ok = false
	return p
}

func (p *Param) SetRegion(region int, v ...float32) {
	util.Argument(len(v) == p.NComp())
	if region == 0 {
		log.Fatal("cannot set parameters in region 0 (vacuum)")
	}
	for c := range v {
		p.cpu[c][region] = v[c]
	}
	p.ok = false
	if p.post_update != nil {
		p.post_update(region)
	}
}

func (p *Param) GetRegion(region int) []float32 {
	v := make([]float32, p.NComp())
	for c := range v {
		v[c] = p.cpu[c][region]
	}
	return v
}

func (p *Param) Gpu() cuda.LUTPtrs {
	if p.ok {
		return p.gpu
	}
	p.upload()
	return p.gpu
}

// XYZ swap here
func (p *Param) upload() {
	log.Println("upload LUT", p.name, p.cpu)
	for c2 := range p.gpu {
		c := util.SwapIndex(c2, p.NComp())
		cu.MemcpyHtoD(cu.DevicePtr(p.gpu[c]), unsafe.Pointer(&p.cpu[c2][0]), cu.SIZEOF_FLOAT32*int64(len(p.cpu[c2])))
	}
	p.ok = true
}

func scale(v []float32, a float32) []float32 {
	for i := range v {
		v[i] *= a
	}
	return v
}

func RegionSetVector(region int, p *Param, v [3]float64) {
	p.SetRegion(region, []float32{float32(v[0]), float32(v[1]), float32(v[2])}...)
}

func init() {
	world.Func("RegionSetVector", RegionSetVector)
}
