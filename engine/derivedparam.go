package engine

type derivedParam struct {
	lut
	updater  func(*derivedParam)
	uptodate bool
}

func (p *derivedParam) init(nComp int, updater func(*derivedParam)) {
	p.lut.init(nComp, p)
	p.updater = updater
}

func (p *derivedParam) invalidate() {
	p.uptodate = false
}

func (p *derivedParam) Cpu() [][NREGION]float32 {
	p.update()
	return p.cpu_buf
}

func (p *derivedParam) update() {
	if !p.uptodate {
		p.updater(p)
		p.gpu_ok = false
		p.uptodate = true
	}
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
