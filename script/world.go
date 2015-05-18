// package script provides a script interpreter for input files and GUI commands.
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
	Identifiers map[string]Expr   // set of defined identifiers
	parent      *scope            // parent scope, if any
	Doc         map[string]string // documentation for identifiers
}

func NewWorld() *World {
	w := new(World)
	w.scope = new(scope)
	w.toplevel = w.scope
	w.toplevel.Doc = make(map[string]string)
	w.LoadStdlib() // loads into toplevel
	return w
}

func (w *scope) init() {
	if w.Identifiers == nil {
		w.Identifiers = make(map[string]Expr)
	}
}

// adds a native variable to the world. E.g.:
// 	var x = 3.14
// 	world.Var("x", &x)
// 	world.MustEval("x") // returns 3.14
func (w *scope) Var(name string, addr interface{}, doc ...string) {
	w.declare(name, newReflectLvalue(addr), doc...)
}

// Hack for fixing the closure caveat:
// Decleare the time variable, the only variable closures close over.
func (w *scope) TVar(name string, addr interface{}, doc ...string) {
	w.declare(name, &TVar{newReflectLvalue(addr)}, doc...)
}

// adds a native variable to the world. It cannot be changed from script.
// 	var x = 3.14
// 	world.ROnly("x", &x)
// 	world.MustEval("x")   // returns 3.14
// 	world.MustExec("x=2") // fails: cannot assign to x
func (w *scope) ROnly(name string, addr interface{}, doc ...string) {
	w.declare(name, newReflectROnly(addr), doc...)
}

// adds a constant. Cannot be changed in any way.
func (w *scope) Const(name string, val interface{}, doc ...string) {
	switch v := val.(type) {
	default:
		panic(fmt.Errorf("const of type %v not handled", typ(v))) // todo: const using reflection
	case float64:
		w.declare(name, floatLit(v), doc...)
	case int:
		w.declare(name, intLit(v), doc...)
	}
}

// adds a special variable to the world. Upon assignment,
// v's Set() will be called.
func (w *scope) LValue(name string, v LValue, doc ...string) {
	w.declare(name, v, doc...)
}

// adds a native function to the world. E.g.:
// 	world.Func("sin", math.Sin)
// 	world.MustEval("sin(0)") // returns 0
func (w *scope) Func(name string, f interface{}, doc ...string) {
	w.declare(name, newFunction(f), doc...)
}

// add identifier but check that it's not declared yet.
func (w *scope) declare(key string, value Expr, doc ...string) {
	if ok := w.safeDeclare(key, value); !ok {
		panic("identifier " + key + " already defined")
	}
	w.document(key, doc...)
}

func (w *scope) safeDeclare(key string, value Expr) (ok bool) {
	w.init()
	lname := strings.ToLower(key)
	if _, ok := w.Identifiers[lname]; ok {
		return false
	}
	w.Identifiers[lname] = value
	return true
}

// resolve identifier in this scope or its parents
func (w *scope) resolve(pos token.Pos, name string) Expr {
	w.init()
	lname := strings.ToLower(name)
	if v, ok := w.Identifiers[lname]; ok {
		return v
	} else {
		if w.parent != nil {
			return w.parent.resolve(pos, name)
		}
		panic(err(pos, "undefined:", name))
	}
}

func (w *World) Resolve(identifier string) (e Expr) {
	defer func() {
		err := recover()
		if err != nil {
			e = nil // not found
		}
	}()
	e = w.toplevel.resolve(0, identifier)
	return
}

// add documentation for identifier
func (w *scope) document(ident string, doc ...string) {
	if w.Doc != nil { // means we want doc for this scope (toplevel only)
		switch len(doc) {
		default:
			panic("too many doc strings for " + ident)
		case 0:
			w.Doc[ident] = ""
		case 1:
			w.Doc[ident] = doc[0]
		}
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
