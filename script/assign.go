package script

import (
	"fmt"
)

func (p *Parser) newAssign(name string, rhs expr) assignment {
	return assignment{p.getvar(name), rhs}
}

type assignment struct {
	left  variable
	right expr
}

func (e assignment) eval() interface{} {
	e.left.assign(e.right) // no eval here
	return nil
}

func (e assignment) String() string {
	return fmt.Sprint(e.left, "=", e.right)
}
