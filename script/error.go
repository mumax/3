package script

import (
	"fmt"
	"go/ast"
	"reflect"
)

func notAllowed(n ast.Node) error {
	return newCompileErr("not allowed: ", reflect.TypeOf(n))
}

type compileErr string

func (c *compileErr) Error() string {
	return string(*c)
}

func newCompileErr(msg ...interface{}) *compileErr {
	e := compileErr(fmt.Sprint(msg...))
	return &e
}
