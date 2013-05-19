package script

import (
	"fmt"
	"go/ast"
	"go/parser"
)

func (w *World) MustCompileExpr(expr string) expr {
	// parse
	tree, e := parser.ParseExpr(expr)
	if e != nil {
		panic(err(fmt.Sprint(e)))
	}
	ast.Print(nil, tree)
	return w.compileExpr(tree)
}

func (w *World) CompileExpr(src string) (code expr, e error) {
	//compile errors are thrown, caught here and returned
	defer func() {
		err := recover()
		if er, ok := err.(*compileErr); ok {
			code = nil
			e = er
		} else {
			panic(err)
		}
	}()
	code = w.MustCompileExpr(src)
	return
}

// compiles a number of statements. E.g.:
// 	x=1
// 	y=2
func (w *World) Compile(src string) (code stmt, e error) {
	//compile errors are thrown, caught here and returned
	defer func() {
		err := recover()
		if er, ok := err.(*compileErr); ok {
			code = nil
			e = er
		} else {
			panic(err)
		}
	}()
	code = w.MustCompile(src)
	return
}

// Compile, with panic on error
func (w *World) MustCompile(src string) (code stmt) {
	fmt.Println("compile:", src)

	// parse
	expr := "func(){" + src + "\n}" // wrap in func to turn into expression
	tree, e := parser.ParseExpr(expr)
	if e != nil {
		panic(err(fmt.Sprint(e)))
	}

	stmts := tree.(*ast.FuncLit).Body.List // strip func again
	ast.Print(nil, stmts)

	block := new(blockStmt)
	for _, s := range stmts {
		block.append(w.compileStmt(s))
	}
	code = block

	return code
}
