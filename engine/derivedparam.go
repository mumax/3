package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
)

type derivedParam struct {
	gpuTable
	cpuTable
	updater  func(*derivedParam)
	uptodate bool
}

func (p *derivedParam) init(nComp int, updater func(*derivedParam)) {
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.updater = updater
}

func (p *derivedParam) gpu() cuda.LUTPtrs {
	if !p.gpu_ok {
		p.upload(p.Cpu())
	}
	return p.gpu_buf
}

func (p *derivedParam) NComp() int { return len(p.cpu_buf) }

func (p *derivedParam) gpu1() cuda.LUTPtr {
	util.Assert(p.NComp() == 1)
	return cuda.LUTPtr(p.gpu()[0])
}

func (p *derivedParam) Cpu() [][NREGION]float32 {
	if !p.uptodate {
		p.updater(p)
		p.gpu_ok = false
	}
	return p.cpu_buf
}

func isZero(v [][NREGION]float32) bool {
	for c := range v {
		for i := range v[c] { // TODO: regions.maxreg
			if v[c][i] != 0 {
				return false
			}
		}
	}
	return true
}
