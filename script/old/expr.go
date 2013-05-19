package script

type Expr interface {
	Eval() interface{}
}
