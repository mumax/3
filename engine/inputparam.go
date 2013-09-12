package engine

//import (
//	"code.google.com/p/mx3/cuda"
//	"code.google.com/p/mx3/data"
//	"code.google.com/p/mx3/util"
//	"log"
//)

type inputParam struct {
	cpu_buf   [][NREGION]float32 // look-up table source
	upd_reg   [NREGION]func() []float64
	timestamp float64
	uptodate  bool
}

func (p *inputParam) init(nComp int) {
	p.cpu_buf = make([][NREGION]float32)
}

func (p *inputParam) Cpu() [][NREGION]float64 {
	if p.timestamp != Time || p.uptodate == false {

		for r := 0; r < regions.maxreg; r++ {
			updFunc := p.upd_reg[r]
			if updFunc == nil {
				continue
			}
			v := updFunc[r]()
			for c := range p.cpu_buf {
				p.cpu_buf[c][r] = float32(v[c])
			}
			// !! p.gpu_ok = false // !!
		}

		p.timestamp = Time
		p.uptodate = true
	}
	return p.cpu_buf
}

func (p *inputParam) setRegion(region int, v []float64) {
	util.Argument(len(v) == len(p.cpu_buf))

	p.upd_reg[region] = nil

	for c := range v {
		p.cpu_buf[c][region] = float32(v[c])
	}
	p.uptodate = false
}

//
//
//// set in all regions except 0
//// TODO: should region zero really have unset params?
//// TODO: check if we always start from 1
//func (p *inputParam) setUniform(v ...float64) {
//	for r := 1; r < NREGION; r++ {
//		p.setRegion(r, v...)
//	}
//}
//
//func (p *inputParam) getRegion(region int) []float64 {
//	cpu, _ := p.Cpu()
//	v := make([]float64, p.nComp)
//	for i := range v {
//		v[i] = float64(cpu[i][region])
//	}
//	return v
//}
//
//func (p *inputParam) GetVec() []float64 {
//	return p.getRegion(1) // TODO: revise
//}
//
//func (p *param) getUniform() []float64 {
//	cpu, _ := p.Cpu()
//	v := make([]float64, p.NComp())
//	for c := range v {
//		x := cpu[c][1]
//		for r := 2; r < regions.maxreg; r++ {
//			if p.cpu_buf[c][r] != x {
//				log.Panicf("%v is not uniform", p.name)
//			}
//		}
//		v[c] = float64(x)
//	}
//	return v
//}
//
//func (p *param) Get() (*data.Slice, bool) {
//	gpu := p.gpu()
//	b := cuda.GetBuffer(p.NComp(), p.Mesh())
//	for c := 0; c < p.NComp(); c++ {
//		cuda.RegionDecode(b.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
//	}
//	return b, true
//}
