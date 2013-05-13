package script

import (
	"fmt"
	"reflect"
	"strings"
)

type world struct {
	identifiers map[string]interface{} // set of defined identifiers
}

func (w *world) init() {
	w.identifiers = make(map[string]interface{})
}

// add identifier but check that it's not declared yet.
func (w *world) declare(key string, value interface{}) {
	lname := strings.ToLower(key)
	if _, ok := w.identifiers[lname]; ok {
		panic("identifier " + key + " already defined")
	}
	w.identifiers[lname] = value
}

func (w *world) AddVar(name string, v Variable) {
	w.declare(name, v)
}

func (p *Parser) get(name string) interface{} {
	lname := strings.ToLower(name)
	if v, ok := p.identifiers[lname]; ok {
		return v
	} else {
		panic(fmt.Errorf("line %v: undefined: %v", p.Line, name))
	}
}

func (w *world) AddFunc(name string, f interface{}) {
	v := reflect.ValueOf(f)
	if v.Kind() != reflect.Func {
		panic(fmt.Errorf("addfunc: expect func, got: %v", reflect.TypeOf(f)))
	}
	w.declare(name, v) // TODO: wrap in Func type?
}

// TODO: add const, like pi, mu0, ...

// TODO: rm
func (w *world) AddFloat(name string, addr *float64) {
	lname := strings.ToLower(name)
	w.AddVar(lname, float{addr})
}
