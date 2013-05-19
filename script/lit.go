package script

import (
	"go/ast"
	"go/token"
	"reflect"
	"strconv"
)

// compiles a basic literal, like numbers and strings
func (w *World) compileBasicLit(n *ast.BasicLit) expr {
	switch n.Kind {
	default:
		panic(err("not allowed:", n.Value, "(", typ(n), ")"))
	case token.INT, token.FLOAT:
		return floatLit(parseFloat(n.Value))
	}
}

type floatLit float64

func (l floatLit) Eval() interface{}  { return float64(l) }
func (l floatLit) Type() reflect.Type { return float64_t }

func parseFloat(str string) float64 {
	v, err := strconv.ParseFloat(str, 64)
	if err != nil {
		panic("internal error") // we were sure it was a number...
	}
	return v
}
