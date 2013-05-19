package script

import (
	"fmt"
	"go/ast"
	"go/parser"
	"reflect"
)

type World struct {
}

func NewWorld() *World {
	return &World{}
}

func (w *World) Compile(src string) (c *code, e error) {
	fmt.Println("compile:", src)

	expr := `func(){` + src + `}` // turn into expression
	tree, err := parser.ParseExpr(expr)
	if err != nil {
		return nil, err
	}

	stmts := tree.(*ast.FuncLit).Body.List
	ast.Print(nil, stmts)

	defer func() {
		err := recover()
		if er, ok := err.(*compileErr); ok {
			c = nil
			e = er
		} else {
			panic(err)
		}
	}()

	for _, s := range stmts {
		c.append(w.compileStmt(s))
	}

	return c, nil
}

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

type code struct {
	list []*stmt
}

func (c *code) append(s *stmt) {
	c.list = append(c.list, s)
}

type stmt struct {
}

func (w *World) compileStmt(st ast.Stmt) *stmt {
	switch st.(type) {
	default:
		panic(notAllowed(st))
	}
}
