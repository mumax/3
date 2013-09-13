package engine

// parameter derived from others (not directly settable)
type derivedParam struct {
	lut
	updater  func(*derivedParam)
	uptodate bool // cleared if parents' value change
}

func (p *derivedParam) init(nComp int, updater func(*derivedParam)) {
	p.lut.init(nComp, p)
	p.updater = updater
}

func (p *derivedParam) invalidate() {
	p.uptodate = false
}

func (p *derivedParam) update() {
	if !p.uptodate {
		p.updater(p)
		p.gpu_ok = false
		p.uptodate = true
	}
}
