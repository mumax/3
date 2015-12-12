package engine

// parameter derived from others (not directly settable). E.g.: Bsat derived from Msat
type derivedInput struct {
	lut                          // GPU storage
	updater  func(*derivedInput) // called to update my value
	uptodate bool                // cleared if parents' value change
	parents  []updater           // parents updated before I'm updated
}

func (p *derivedInput) init(nComp int, parents []updater, updater func(*derivedInput)) {
	p.lut.init(nComp, p) // pass myself to update me if needed
	p.updater = updater
	p.parents = parents
}

func (p *derivedInput) invalidate() {
	p.uptodate = false
}

func (p *derivedInput) update() {
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
func (p *derivedInput) GetRegion(r int) []float64 {
	lut := p.cpuLUT() // updates me if needed
	v := make([]float64, p.NComp())
	for c := range v {
		v[c] = float64(lut[c][r])
	}
	return v
}
