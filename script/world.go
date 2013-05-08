package script

import "fmt"

type world struct {
	variables map[string]Variable
}

func (w *world) init() {
	w.variables = make(map[string]Variable)
}

func (w *world) AddVar(name string, v Variable) {
	if _, ok := w.variables[name]; ok {
		panic("variable " + name + " already defined")
	}
	w.variables[name] = v
}

func (p *Parser) getvar(name string) Variable {
	if v, ok := p.variables[name]; ok {
		return v
	} else {
		panic(fmt.Errorf("line %v: undefined: %v", p.scan.Pos().Line, name))
	}
}

func (w *world) AddFloat(name string, addr *float64) {
	w.AddVar(name, float{addr})
}
