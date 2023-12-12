package script

// Here be dragons

import (
	"github.com/mumax/3/v3/data"
	"reflect"
)

type ScalarFunction interface {
	Expr
	Float() float64
}

// converts float64 to ScalarFunction
type scalFn struct{ in Expr }

func (c *scalFn) Eval() interface{}  { return c }
func (c *scalFn) Type() reflect.Type { return ScalarFunction_t }
func (c *scalFn) Float() float64     { return c.in.Eval().(float64) }
func (c *scalFn) Child() []Expr      { return []Expr{c.in} }
func (c *scalFn) Fix() Expr          { return &scalFn{in: c.in.Fix()} }

type VectorFunction interface {
	Expr
	Float3() data.Vector
}

// converts data.Vector to VectorFunction
type vecFn struct{ in Expr }

func (c *vecFn) Eval() interface{}   { return c }
func (c *vecFn) Type() reflect.Type  { return VectorFunction_t }
func (c *vecFn) Float3() data.Vector { return c.in.Eval().(data.Vector) }
func (c *vecFn) Child() []Expr       { return []Expr{c.in} }
func (c *vecFn) Fix() Expr           { return &vecFn{in: c.in.Fix()} }
