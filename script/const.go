package script

import "reflect"

type Const struct {
	value interface{}
	typ   reflect.Type
}

func NewConst(e Expr) *Const {
	return &Const{value: e.Eval(), typ: e.Type()}
}

func (c *Const) Eval() interface{}  { return c.value }
func (c *Const) Type() reflect.Type { return c.typ }
func (c *Const) Child() []Expr      { return nil }
func (c *Const) Fix() Expr          { return c }
