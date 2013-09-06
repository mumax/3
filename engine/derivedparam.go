package engine

type derivedParam struct {
	param
	update func(*param)
}

func (p *derivedParam) init(nComp int, name, unit string, update func(*param)) {
	p.init_(nComp, name, unit)
	p.update = update
}

func (p *derivedParam) Cpu() [][NREGION]float32 {
	p.update(&p.param)
	return p.cpu
}
