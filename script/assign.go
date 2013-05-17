package script

import (
	"fmt"
)

type assignment struct {
	left  Variable
	right Expr
}

func (p *Parser) newAssign(name string, rhs Expr) assignment {
	if v, ok := p.get(name).(Variable); ok {
		return assignment{v, rhs}
	}
	panic(fmt.Errorf("line %v: cannot assign to %v", p.Position, name))
}

func (e assignment) Eval() interface{} {
	e.left.Assign(e.right) // no eval here
	return nil
}

func (e assignment) String() string {
	return fmt.Sprint(e.left, "=", e.right)
}
