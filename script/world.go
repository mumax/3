package script

type world struct {
	variables map[string]variable
}

func (w *world) init() {
	w.variables = make(map[string]variable)
}

func (w *world) addvar(name string, v variable) {
	if _, ok := w.variables[name]; ok {
		panic("variable " + name + " already defined")
	}
	w.variables[name] = v
}

func (p *parser) getvar(name string) variable {
	if v, ok := p.variables[name]; ok {
		return v
	} else {
		panic(p.undefined())
	}
}

func (w *world) addFloat(name string, addr *float64) {
	w.addvar(name, float{addr})
}
