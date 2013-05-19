package script

import (
	"fmt"
	"go/ast"
	"go/parser"
	"reflect"
)

func (w *World) Compile(src string) (code stmt, e error) {
	fmt.Println("compile:", src)

	// parse
	expr := "func(){" + src + "\n}" // wrap in func to turn into expression
	tree, err := parser.ParseExpr(expr)
	if err != nil {
		return nil, err
	}

	// compile errors are thrown, caught here and returned
	//	defer func() {
	//		err := recover()
	//		if er, ok := err.(*compileErr); ok {
	//			c = nil
	//			e = er
	//		} else {
	//			panic(err)
	//		}
	//	}()

	stmts := tree.(*ast.FuncLit).Body.List // strip func again
	ast.Print(nil, stmts)

	block := new(blockStmt)
	for _, s := range stmts {
		block.append(w.compileStmt(s))
	}
	code = block

	return code, nil
}

func typ(i interface{}) string {
	return reflect.TypeOf(reflect.ValueOf(i).Interface()).String()
}
