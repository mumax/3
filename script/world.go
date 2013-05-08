package script

import (
	"fmt"
	"reflect"
)

type world struct {
	identifiers map[string]struct{} // set of defined identifiers
	variables   map[string]Variable
	functions   map[string]reflect.Value
}

func (w *world) init() {
	w.identifiers = make(map[string]struct{})
	w.variables = make(map[string]Variable)
	w.functions = make(map[string]reflect.Value)
}

// checks that name has not yet been declared,
// then adds it to the known identifiers so it can not be declared a second time.
func (w *world) declare(name string) {
	if _, ok := w.identifiers[name]; ok {
		panic("identifier " + name + " already defined")
	}
	w.identifiers[name] = struct{}{}
}

func (w *world) AddVar(name string, v Variable) {
	w.declare(name)
	w.variables[name] = v
}

func (p *Parser) getvar(name string) Variable {
	if v, ok := p.variables[name]; ok {
		return v
	} else {
		panic(fmt.Errorf("line %v: undefined: %v", p.scan.Pos().Line, name))
	}
}

func (w *world) AddFunc(name string, f interface{}) {
	w.declare(name)
	v := reflect.ValueOf(f)
	if v.Kind() != reflect.Func {
		panic(fmt.Errorf("addfunc: expect func, got: %v", reflect.TypeOf(f)))
	}
	w.functions[name] = v
}

func (p *Parser) getfunc(name string) reflect.Value {
	if v, ok := p.functions[name]; ok {
		return v
	} else {
		panic(fmt.Errorf("line %v: undefined: %v", p.scan.Pos().Line, name))
	}
}

// TODO: add const, like pi, mu0, ...

// TODO: rm
func (w *world) AddFloat(name string, addr *float64) {
	w.AddVar(name, float{addr})
}
