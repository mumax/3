package script

import (
	"go/ast"
	"reflect"
)

// left-hand value in (single) assign statement
type LValue interface {
	Expr                  // evalutes
	SetValue(interface{}) // assigns a new value
}

func (w *World) compileLvalue(lhs ast.Node) LValue {
	switch concrete := lhs.(type) {
	default:
		panic(err(lhs.Pos(), "cannot assign to", typ(lhs)))
	case *ast.Ident:
		if l, ok := w.resolve(lhs.Pos(), concrete.Name).(LValue); ok {
			return l
		} else {
			panic(err(lhs.Pos(), "cannot assign to", concrete.Name))
		}
	}
}

// read-only value (from script, but mutable from outside)
type reflectROnly struct {
	elem reflect.Value
}

func newReflectROnly(addr interface{}) *reflectROnly {
	elem := reflect.ValueOf(addr)
	if elem.Kind() == 0 {
		panic("variable/constant needs to be passed as pointer to addressable value")
	}
	return &reflectROnly{elem}
}

func (l *reflectROnly) Eval() interface{} {
	return l.elem.Interface()
}

func (l *reflectROnly) Type() reflect.Type {
	return l.elem.Type()
}

// general lvalue implementation using reflect.
// lhs must be settable, e.g. address of something:
// 	var x float64
// 	newReflectLValue(&x)
func newReflectLvalue(addr interface{}) LValue {
	elem := reflect.ValueOf(addr).Elem()
	if elem.Kind() == 0 {
		panic("variable/constant needs to be passed as pointer to addressable value")
	}
	return &reflectLvalue{reflectROnly{elem}}
}

type reflectLvalue struct {
	reflectROnly
}

func (l *reflectLvalue) SetValue(rvalue interface{}) {
	l.elem.Set(reflect.ValueOf(rvalue))
}
