package script

import (
	"reflect"
)

// left-hand value in (single) assign statement
type lvalue interface {
	expr             // evalutes
	Set(interface{}) // assigns a new value
}

// general lvalue implementation using reflect.
// lhs must be settable, e.g. address of something
func newReflectLvalue(lhs interface{}) lvalue {
	return &reflectLvalue{reflect.ValueOf(lhs).Elem()}
}

type reflectLvalue struct {
	elem reflect.Value
}

func (l *reflectLvalue) Set(rvalue interface{}) {
	l.elem.Set(reflect.ValueOf(rvalue))
}

func (l *reflectLvalue) Eval() interface{} {
	return l.elem.Interface()
}

func (l *reflectLvalue) Type() reflect.Type {
	return l.elem.Type()
}
