package script

import (
	"fmt"
)

func (p *Parser) newAssign(name string, rhs Expr) assignment {
	return assignment{p.getvar(name), rhs}
}

type assignment struct {
	left  Variable
	right Expr
}

func (e assignment) Eval() interface{} {
	e.left.Assign(e.right) // no eval here
	return nil
}

func (e assignment) String() string {
	return fmt.Sprint(e.left, "=", e.right)
}
