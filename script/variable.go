package script

import (
	"fmt"
)

type variable interface {
	expr
	assign(expr)
}

type float struct {
	addr *float64
}

func (f float) String() string {
	return fmt.Sprintf("float@%p", f.addr)
}

func (f float) eval() interface{} {
	return *(f.addr)
}

func (f float) assign(rhs expr) {
	*(f.addr) = rhs.eval().(float64)
}
