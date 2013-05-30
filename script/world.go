package script

import (
	"fmt"
	"go/token"
	"strings"
)

// World stores an interpreted program's state
// like declared variables and functions.
type World struct {
	*scope
	toplevel *scope
}

// scope stores identifiers
type scope struct {
	identifiers map[string]Expr // set of defined identifiers
	parent      *scope          // parent scope, if any
}

func NewWorld() *World {
	w := new(World)
	w.scope = new(scope)
	w.toplevel = w.scope
	w.LoadStdlib() // loads into toplevel
	return w
}

func (w *scope) init() {
	if w.identifiers == nil {
		w.identifiers = make(map[string]Expr)
	}
}

// adds a native variable to the world. E.g.:
// 	var x = 3.14
// 	world.Var("x", &x)
// 	world.MustEval("x") // returns 3.14
func (w *scope) Var(name string, addr interface{}) {
	w.declare(name, newReflectLvalue(addr))
}

// adds a native variable to the world. It cannot be changed from script.
// 	var x = 3.14
// 	world.ROnly("x", &x)
// 	world.MustEval("x")   // returns 3.14
// 	world.MustExec("x=2") // fails: cannot assign to x
func (w *scope) ROnly(name string, addr interface{}) {
	w.declare(name, newReflectROnly(addr))
}

// adds a constant. Cannot be changed in any way.
func (w *scope) Const(name string, val interface{}) {
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
func (w *scope) LValue(name string, v LValue) {
	w.declare(name, v)
}

// adds a native function to the world. E.g.:
// 	world.Func("sin", math.Sin)
// 	world.MustEval("sin(0)") // returns 0
func (w *scope) Func(name string, f interface{}) {
	// TODO: specialize for float64 funcs etc
	w.declare(name, newReflectFunc(f))
}

// add identifier but check that it's not declared yet.
func (w *scope) declare(key string, value Expr) {
	w.init()
	lname := strings.ToLower(key)
	if _, ok := w.identifiers[lname]; ok {
		panic("identifier " + key + " already defined")
	}
	w.identifiers[lname] = value
}

// resolve identifier in this scope or its parents
func (w *scope) resolve(pos token.Pos, name string) Expr {
	w.init()
	lname := strings.ToLower(name)
	if v, ok := w.identifiers[lname]; ok {
		return v
	} else {
		if w.parent != nil {
			return w.parent.resolve(pos, name)
		}
		panic(err(pos, "undefined:", name))
	}
}

func (w *World) EnterScope() {
	par := w.scope
	w.scope = new(scope)
	w.scope.parent = par
}

func (w *World) ExitScope() {
	w.scope = w.scope.parent
	if w.scope == nil { // went above toplevel
		panic("bug")
	}
}
