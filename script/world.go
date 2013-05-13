package script

import (
	"fmt"
	"reflect"
	"strings"
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
	name = strings.ToLower(name)
	w.declare(name)
	w.variables[name] = v
}

func (p *Parser) getvar(name string) Variable {
	lname := strings.ToLower(name)
	if v, ok := p.variables[lname]; ok {
		return v
	} else {
		panic(fmt.Errorf("line %v: undefined: %v", p.Line, name))
	}
}

func (w *world) AddFunc(name string, f interface{}) {
	lname := strings.ToLower(name)
	w.declare(name)
	v := reflect.ValueOf(f)
	if v.Kind() != reflect.Func {
		panic(fmt.Errorf("addfunc: expect func, got: %v", reflect.TypeOf(f)))
	}
	w.functions[lname] = v
}

func (p *Parser) getfunc(name string) reflect.Value {
	lname := strings.ToLower(name)
	if v, ok := p.functions[lname]; ok {
		return v
	} else {
		panic(fmt.Errorf("line %v: undefined: %v", p.Line, name))
	}
}

// TODO: add const, like pi, mu0, ...

// TODO: rm
func (w *world) AddFloat(name string, addr *float64) {
	lname := strings.ToLower(name)
	w.AddVar(lname, float{addr})
}
