package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
	"math"
)

type inputParam struct {
	param
	cpu_stamp float64 // timestamp for cpu data
	upd_reg   [NREGION]func() []float64
}

func (p *inputParam) init(nComp int, name, unit string) {
	p.param.init_(nComp, name, unit, p.update)
	p.cpu_stamp = math.Inf(-1)
}

func (p *inputParam) update() {
	if p.cpu_stamp != Time {
		log.Println("update", p.Name())
		p.cpu_stamp = Time

		for r := 0; r < regions.maxreg; r++ {
			if p.upd_reg[r] != nil {
				p.gpu_ok = false
				v := p.upd_reg[r]()
				for c := range p.cpu_buf {
					p.cpu_buf[c][r] = float32(v[c])
				}
			}
		}
	}
}

func (p *inputParam) timestamp() float64 {
	p.update()
	return p.cpu_stamp
}

func (p *inputParam) setRegion(region int, v ...float64) {
	util.Argument(len(v) == p.NComp()) // note: also panics if param not initialized (nComp = 0)

	p.gpu_ok = false
	p.upd_reg[region] = nil
	for c := range v {
		p.cpu_buf[c][region] = float32(v[c])
	}
}

// set in all regions except 0
func (p *inputParam) setUniform(v ...float64) {
	for r := 1; r < NREGION; r++ {
		p.setRegion(r, v...)
	}
}

func (p *inputParam) getRegion(region int) []float64 {
	cpu := p.Cpu()
	v := make([]float64, p.nComp)
	for i := range v {
		v[i] = float64(cpu[i][region])
	}
	return v
}

func (p *inputParam) GetVec() []float64 {
	return p.getRegion(1) // TODO: revise
}

func (p *param) getUniform() []float64 {
	cpu := p.Cpu()
	v := make([]float64, p.NComp())
	for c := range v {
		x := cpu[c][1]
		for r := 2; r < regions.maxreg; r++ {
			if p.cpu_buf[c][r] != x {
				log.Panicf("%v is not uniform", p.name)
			}
		}
		v[c] = float64(x)
	}
	return v
}

func (p *param) Get() (*data.Slice, bool) {
	gpu := p.gpu()
	b := cuda.GetBuffer(p.NComp(), p.Mesh())
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
	}
	return b, true
}
