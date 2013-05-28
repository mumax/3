package script

import (
	"fmt"
	"go/token"
	"strings"
)

// World stores an interpreted program's state
// like declared variables and functions.
type World struct {
	identifiers map[string]Expr // set of defined identifiers
	Debug       bool            // print debug info?
}

func NewWorld() *World {
	w := new(World)
	w.init()
	w.Debug = false
	w.LoadStdlib()
	return w
}

func (w *World) init() {
	if w.identifiers == nil {
		w.identifiers = make(map[string]Expr)
	}
}

// adds a native variable to the world. E.g.:
// 	var x = 3.14
// 	world.Var("x", &x)
// 	world.MustEval("x") // returns 3.14
func (w *World) Var(name string, addr interface{}) {
	w.declare(name, newReflectLvalue(addr))
}

// adds a native variable to the world. It cannot be changed from script.
// 	var x = 3.14
// 	world.ROnly("x", &x)
// 	world.MustEval("x")   // returns 3.14
// 	world.MustExec("x=2") // fails: cannot assign to x
func (w *World) ROnly(name string, addr interface{}) {
	w.declare(name, newReflectROnly(addr))
}

// adds a constant. Cannot be changed in any way.
func (w *World) Const(name string, val interface{}) {
	switch v := val.(type) {
	default:
		panic(fmt.Errorf("const of type %v not handled", typ(v))) // todo: const using reflection
	case float64:
		w.declare(name, floatLit(v))
	case int:
		w.declare(name, intLit(v))
	}
}

// adds a special variable to the world. Upon assignment,
// v's Set() will be called.
func (w *World) LValue(name string, v LValue) {
	w.declare(name, v)
}

// adds a native function to the world. E.g.:
// 	world.Func("sin", math.Sin)
// 	world.MustEval("sin(0)") // returns 0
func (w *World) Func(name string, f interface{}) {
	// TODO: specialize for float64 funcs etc
	w.declare(name, newReflectFunc(f))
}

// add identifier but check that it's not declared yet.
func (w *World) declare(key string, value Expr) {
	w.init()
	lname := strings.ToLower(key)
	if _, ok := w.identifiers[lname]; ok {
		panic("identifier " + key + " already defined")
	}
	w.identifiers[lname] = value
}

func (w *World) resolve(pos token.Pos, name string) Expr {
	w.init()
	lname := strings.ToLower(name)
	if v, ok := w.identifiers[lname]; ok {
		return v
	} else {
		panic(err(pos, "undefined:", name)) // TODO: add pos
	}
}
