package script

import (
	"go/ast"
)

type stmt interface {
}

func (w *World) compileStmt(st ast.Stmt) stmt {
	switch concrete := st.(type) {
	default:
		panic(newCompileErr("not allowed:", st))
	case *ast.AssignStmt:
		return w.compileAssignStmt(concrete)
	}
}
