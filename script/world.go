package script

import (
	"strings"
)

type World struct {
	identifiers map[string]expr // set of defined identifiers
}

func NewWorld() *World {
	w := new(World)
	w.init()
	return w
}

// adds a native variable to the world. E.g.:
// 	var x float64
// 	world.Var("x", &x)
func (w *World) Var(name string, addr interface{}) {
	w.declare(name, newReflectLvalue(addr))
}

func (w *World) init() {
	if w.identifiers == nil {
		w.identifiers = make(map[string]expr)
	}
}

// add identifier but check that it's not declared yet.
func (w *World) declare(key string, value expr) {
	w.init()
	lname := strings.ToLower(key)
	if _, ok := w.identifiers[lname]; ok {
		panic(err("identifier " + key + " already defined"))
	}
	w.identifiers[lname] = value
}

func (w *World) resolve(name string) expr {
	w.init()
	lname := strings.ToLower(name)
	if v, ok := w.identifiers[lname]; ok {
		return v
	} else {
		panic(err("undefined:", name))
	}
}
