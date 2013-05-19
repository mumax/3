package script

import (
	"strings"
)

type World struct {
	identifiers map[string]interface{} // set of defined identifiers
}

func NewWorld() *World {
	w := new(World)
	w.init()
	return w
}

func (w *World) init() {
	if w.identifiers == nil {
		w.identifiers = make(map[string]interface{})
	}
}

// add identifier but check that it's not declared yet.
func (w *World) declare(key string, value interface{}) {
	w.init()
	lname := strings.ToLower(key)
	if _, ok := w.identifiers[lname]; ok {
		panic(newCompileErr("identifier " + key + " already defined"))
	}
	w.identifiers[lname] = value
}

func (w *World) resolve(name string) interface{} {
	w.init()
	lname := strings.ToLower(name)
	if v, ok := w.identifiers[lname]; ok {
		return v
	} else {
		panic(newCompileErr("undefined:", name))
	}
}
