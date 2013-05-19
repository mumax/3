package script

import (
	"fmt"
)

type Variable interface {
	Expr
	Assign(Expr)
}

type float struct {
	addr *float64
}

func (f float) String() string {
	return fmt.Sprintf("float@%p", f.addr)
}

func (f float) Eval() interface{} {
	return *(f.addr)
}

func (f float) Assign(rhs Expr) {
	*(f.addr) = rhs.Eval().(float64)
}
