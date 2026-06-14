package script

import "reflect"

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

func (l *reflectROnly) Eval() interface{}  { return l.elem.Interface() }
func (l *reflectROnly) Type() reflect.Type { return l.elem.Type() }
func (l *reflectROnly) Child() []Expr      { return nil }
func (l *reflectROnly) Fix() Expr          { return NewConst(l) }
