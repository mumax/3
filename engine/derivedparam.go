package engine

// parameter derived from others (not directly settable). E.g.: Bsat derived from Msat
type derivedParam struct {
	lut                          // GPU storage
	updater  func(*derivedParam) // called to update my value
	uptodate bool                // cleared if parents' value change
	parents  []updater           // parents updated before I'm updated
}

func (p *derivedParam) init(nComp int, parents []updater, updater func(*derivedParam)) {
	p.lut.init(nComp, p) // pass myself to update me if needed
	p.updater = updater
	p.parents = parents
}

func (p *derivedParam) invalidate() {
	p.uptodate = false
}

func (p *derivedParam) update() {
	for _, par := range p.parents {
		par.update() // may invalidate me
	}
	if !p.uptodate {
		p.updater(p)
		p.gpu_ok = false
		p.uptodate = true
	}
}

// Get value in region r.
func (p *derivedParam) GetRegion(r int) []float64 {
	lut := p.cpuLUT() // updates me if needed
	v := make([]float64, p.NComp())
	for c := range v {
		v[c] = float64(lut[c][r])
	}
	return v
}
