package engine

import ()

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

func (p *derivedParam) invalidate() {
	p.uptodate = false
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
