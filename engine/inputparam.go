package engine

import (
	"code.google.com/p/mx3/util"
	"log"
)

// input parameter, settable by user
type inputParam struct {
	lut
	upd_reg   [NREGION]func() []float64 // time-dependent values
	timestamp float64                   // used not to double-evaluate f(t)
	children  []derived                 // derived parameters
	descr
}

type derived interface {
	invalidate()
}

func (p *inputParam) init(nComp int, name, unit string, children []derived) {
	p.lut.init(nComp, p)
	p.descr = descr{name, unit}
	p.children = children
}

func (p *inputParam) update() {
	if p.timestamp != Time {
		changed := false
		// update functions of time
		for r := 0; r < NREGION; r++ { // TODO: 1..maxreg
			updFunc := p.upd_reg[r]
			if updFunc != nil {
				p.bufset_(r, updFunc())
				changed = true
			}
		}
		p.timestamp = Time
		if changed {
			p.invalidate()
		}
	}
}

// set in one region
func (p *inputParam) setRegion(region int, v []float64) {
	p.setRegions(region, region+1, v)
}

// set in all regions except 0
// TODO: should region zero really have unset params?
// TODO: check if we always start from 1
func (p *inputParam) setUniform(v []float64) {
	p.setRegions(1, NREGION, v)
}

// set in regions r1..r2(excl)
func (p *inputParam) setRegions(r1, r2 int, v []float64) {
	util.Argument(len(v) == len(p.cpu_buf))
	for r := r1; r < r2; r++ {
		p.upd_reg[r] = nil
		p.bufset_(r, v)
	}
	p.invalidate()
}

func (p *inputParam) bufset_(region int, v []float64) {
	for c := range p.cpu_buf {
		p.cpu_buf[c][region] = float32(v[c])
	}
}

// mark my GPU copy and my children as invalid (need update)
func (p *inputParam) invalidate() {
	p.gpu_ok = false
	for _, c := range p.children {
		c.invalidate()
	}
}

func (p *inputParam) getRegion(region int) []float64 {
	cpu := p.CpuLUT()
	v := make([]float64, p.NComp())
	for i := range v {
		v[i] = float64(cpu[i][region])
	}
	return v
}

func (p *inputParam) GetVec() []float64 {
	return p.getRegion(1) // TODO: revise
}

func (p *inputParam) getUniform() []float64 {
	v1 := p.getRegion(1)
	cpu := p.CpuLUT()
	for r := 2; r < regions.maxreg; r++ {
		for c := range v1 {
			if cpu[c][r] != float32(v1[c]) {
				log.Panicf("%v is not uniform", p.name)
			}
		}
	}
	return v1
}

func (p *inputParam) Save()           { Save(p) }
func (p *inputParam) SaveAs(f string) { SaveAs(p, f) }
