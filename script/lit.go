package script

import (
	"go/ast"
	"go/token"
	"reflect"
	"strconv"
)

// compiles a basic literal, like numbers and strings
func (w *World) compileBasicLit(n *ast.BasicLit) Expr {
	switch n.Kind {
	default:
		panic(err(n.Pos(), "not allowed:", n.Value, "(", typ(n), ")"))
	case token.FLOAT:
		return floatLit(parseFloat(n.Value))
	case token.INT:
		return intLit(parseInt(n.Value))
	case token.STRING:
		return stringLit(n.Value[1 : len(n.Value)-1]) // remove quotes
	}
}

type floatLit float64

func (l floatLit) Eval() interface{}  { return float64(l) }
func (l floatLit) Type() reflect.Type { return float64_t }
func (l floatLit) Child() []Expr      { return nil }
func (l floatLit) Fix() Expr          { return l }

type intLit int

func (l intLit) Eval() interface{}  { return int(l) }
func (l intLit) Type() reflect.Type { return int_t }
func (l intLit) Child() []Expr      { return nil }
func (l intLit) Fix() Expr          { return l }

type stringLit string

func (l stringLit) Eval() interface{}  { return string(l) }
func (l stringLit) Type() reflect.Type { return string_t }
func (l stringLit) Child() []Expr      { return nil }
func (l stringLit) Fix() Expr          { return l }

type boolLit bool

func (l boolLit) Eval() interface{}  { return bool(l) }
func (l boolLit) Type() reflect.Type { return bool_t }
func (l boolLit) Child() []Expr      { return nil }
func (l boolLit) Fix() Expr          { return l }

func parseFloat(str string) float64 {
	v, err := strconv.ParseFloat(str, 64)
	if err != nil {
		panic("internal error") // we were sure it was a number...
	}
	return v
}

func parseInt(str string) int {
	v, err := strconv.Atoi(str)
	if err != nil {
		panic("internal error") // we were sure it was a number...
	}
	return v
}
