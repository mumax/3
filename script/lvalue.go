package script

import (
	"reflect"
)

// left-hand value in (single) assign statement
type LValue interface {
	Expr                  // evalutes
	SetValue(interface{}) // assigns a new value
}

// general lvalue implementation using reflect.
// lhs must be settable, e.g. address of something:
// 	var x float64
// 	newReflectLValue(&x)
func newReflectLvalue(lhs interface{}) LValue {
	return &reflectLvalue{reflect.ValueOf(lhs).Elem()}
}

type reflectLvalue struct {
	elem reflect.Value
}

func (l *reflectLvalue) SetValue(rvalue interface{}) {
	l.elem.Set(reflect.ValueOf(rvalue))
}

func (l *reflectLvalue) Eval() interface{} {
	return l.elem.Interface()
}

func (l *reflectLvalue) Type() reflect.Type {
	return l.elem.Type()
}
