package engine

// declare functionality for interpreted input scripts

import (
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/script"
	"reflect"
)

func CompileFile(fname string) (*script.BlockStmt, error) {
	bytes, err := httpfs.Read(fname)
	if err != nil {
		return nil, err
	}
	return World.Compile(string(bytes))
}

func Eval(code string) {
	tree, err := World.Compile(code)
	if err != nil {
		LogIn(code)
		LogErr(err.Error())
		return
	}
	LogIn(rmln(tree.Format()))
	tree.Eval()
}

func Eval1Line(code string) interface{} {
	tree, err := World.Compile(code)
	if err != nil {
		LogErr(err.Error())
		return nil
	}
	if len(tree.Children) != 1 {
		LogErr("expected single statement:" + code)
		return nil
	}
	return tree.Children[0].Eval()
}

// holds the script state (variables etc)
var World = script.NewWorld()

// Add a function to the script world
func DeclFunc(name string, f interface{}, doc string) {
	World.Func(name, f, doc)
}

// Add a constant to the script world
func DeclConst(name string, value float64, doc string) {
	World.Const(name, value, doc)
}

// Add a read-only variable to the script world.
// It can be changed, but not by the user.
func DeclROnly(name string, value interface{}, doc string) {
	World.ROnly(name, value, doc)
	GUIAdd(name, value)
}

// Add a (pointer to) variable to the script world
func DeclVar(name string, value interface{}, doc string) {
	World.Var(name, value, doc)
	GUIAdd(name, value)
}

// Hack for fixing the closure caveat:
// Defines "t", the time variable, handled specially by Fix()
func DeclTVar(name string, value interface{}, doc string) {
	World.TVar(name, value, doc)
	GUIAdd(name, value)
}

// Add an LValue to the script world.
// Assign to LValue invokes SetValue()
func DeclLValue(name string, value LValue, doc string) {
	World.LValue(name, newLValueWrapper(value), doc)
	GUIAdd(name, value)
}

// LValue is settable
type LValue interface {
	SetValue(interface{}) // assigns a new value
	Eval() interface{}    // evaluate and return result (nil for void)
	Type() reflect.Type   // type that can be assigned and will be returned by Eval
}

// evaluate code, exit on error (behavior for input files)
func EvalFile(code *script.BlockStmt) {
	for i := range code.Children {
		formatted := rmln(script.Format(code.Node[i]))
		LogIn(formatted)
		code.Children[i].Eval()
	}
}

// wraps LValue and provides empty Child()
type lValueWrapper struct {
	LValue
}

func newLValueWrapper(lv LValue) script.LValue {
	return &lValueWrapper{lv}
}

func (w *lValueWrapper) Child() []script.Expr { return nil }
func (w *lValueWrapper) Fix() script.Expr     { return script.NewConst(w) }

func (w *lValueWrapper) InputType() reflect.Type {
	if i, ok := w.LValue.(interface {
		InputType() reflect.Type
	}); ok {
		return i.InputType()
	} else {
		return w.Type()
	}
}
